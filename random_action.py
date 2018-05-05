#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotationの動作確認として十分に多くの局面で動作に問題がないかを調べる

正解のデータはpython-shogiで生成する
手番側の行動と王手はpython-shogiの機能で正解を生成できる
利きについては成りを除けば自身の駒で移動できない点が差分になる
"""

import os
from enum import Enum
from pathlib import Path

import numpy as np
import tensorflow as tf
import shogi
from shogi import (
    PIECE_TYPES, bit_scan, BB_R45_ATTACKS, BB_L45_ATTACKS, BB_SHIFT_R45,
    BB_SHIFT_L45, BB_RANK_ATTACKS, BB_FILE_ATTACKS, BB_ALL, BB_VOID,
    PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, KING,
    PROM_PAWN, PROM_LANCE, PROM_KNIGHT, PROM_SILVER, PROM_BISHOP, PROM_ROOK,
    BLACK, WHITE
)
from tqdm import trange
from dotenv import load_dotenv
from sklearn.externals import joblib

from annotation import AnnotationLayer
from annotation.piece import Piece
from annotation.direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/3/22'


class Board(object):
    """
    python-shogiのBoardは横型で、駒の持ち主の情報がビットボードに格納されているので、扱いにくい
    自分でBoardを定義する
    """
    def __init__(self, flip):
        # 後手から見た局面を保持するならTrue
        self.flip = flip

        # 見ている盤面の下側をBLACK、上側をWHITEとする
        self.board = self.initialize_board()
        self.black_hand = np.zeros(7, dtype=np.int32)
        self.white_hand = np.zeros_like(self.black_hand)

        # 見ている方で下側の手番ならFalse、上側ならTrue
        # flipの状態では、上側からスタートする
        self.player = flip

    def __str__(self):
        s = """<{flip}>
{board}
{black_hand}
{white_hand}
""".format(flip='reversed' if self.flip else 'positive', board=self.board,
           black_hand=self.black_hand, white_hand=self.white_hand)

        return s

    @staticmethod
    def initialize_board():
        board = np.empty((9, 9), dtype=np.int32)
        board[:] = Piece.EMPTY

        board[:, 2] = Piece.WHITE_FU
        board[:, 6] = Piece.BLACK_FU
        board[1, 1] = Piece.WHITE_KA
        board[7, 7] = Piece.BLACK_KA
        board[7, 1] = Piece.WHITE_HI
        board[1, 7] = Piece.BLACK_HI
        for i, p in enumerate([Piece.BLACK_KY, Piece.BLACK_KE, Piece.BLACK_GI,
                               Piece.BLACK_KI, Piece.BLACK_OU]):
            board[i, 0] = p + 14
            board[8 - i, 0] = p + 14

            board[i, 8] = p
            board[8 - i, 8] = p

        return board

    def step(self, action):
        if action.action_type == ActionType.MOVE:
            if self.flip:
                t = action.reversed_target
                s = action.reversed_source
            else:
                t = action.positive_target
                s = action.positive_source

            target = self.board[t]
            if target != Piece.EMPTY:
                # 駒を取る
                if target >= Piece.WHITE_FU:
                    if target >= Piece.WHITE_TO:
                        target -= Piece.WHITE_TO
                    else:
                        target -= Piece.WHITE_FU
                    self.black_hand[target] += 1

                    # 下側 BLACKの手番
                    assert not self.player
                else:
                    if target >= Piece.BLACK_TO:
                        target -= Piece.BLACK_TO
                    self.white_hand[target] += 1

                    # 上側 WHITEの手番
                    assert self.player

            # 駒を動かす
            self.board[t] = self.board[s]
            self.board[s] = Piece.EMPTY
            if action.promotion:
                self.board[t] += 8
        else:
            if self.flip:
                t = action.reversed_target
            else:
                t = action.positive_target

            if self.player:
                self.white_hand[action.piece] -= 1
                self.board[t] = action.piece + Piece.WHITE_FU
            else:
                self.black_hand[action.piece] -= 1
                self.board[t] = action.piece + Piece.BLACK_FU

        # 手番を次へ渡す
        self.player = not self.player


class ActionType(Enum):
    MOVE = 0
    DROP = 1


class Action(object):
    def __init__(self, usi):
        usi = str(usi)

        if usi[1] == '*':
            self.action_type = ActionType.DROP

            # 先後の情報がないので、BLACKにしておく
            tmp = {
                'P': Piece.BLACK_FU, 'L': Piece.BLACK_KY, 'N': Piece.BLACK_KE,
                'S': Piece.BLACK_GI, 'G': Piece.BLACK_KI, 'B': Piece.BLACK_KA,
                'R': Piece.BLACK_HI
            }
            self.piece = tmp[usi[0]]
            self.source = (9, 9)
            self.target = (int(usi[2]) - 1, ord(usi[3]) - ord('a'))
        else:
            self.action_type = ActionType.MOVE

            self.promotion = len(usi) == 5
            self.source = (int(usi[0]) - 1, ord(usi[1]) - ord('a'))
            self.target = (int(usi[2]) - 1, ord(usi[3]) - ord('a'))

            if self.source[0] == self.target[0]:
                if self.source[1] > self.target[1]:
                    self.direction = Direction.UP
                else:
                    self.direction = Direction.DOWN
                self.distance = abs(self.source[1] - self.target[1])
            elif self.source[1] == self.target[1]:
                if self.source[0] > self.target[0]:
                    self.direction = Direction.RIGHT
                else:
                    self.direction = Direction.LEFT
                self.distance = abs(self.source[0] - self.target[0])
            elif (self.source[0] - self.source[1] ==
                    self.target[0] - self.target[1]):
                if self.source[0] > self.target[0]:
                    self.direction = Direction.RIGHT_UP
                else:
                    self.direction = Direction.LEFT_DOWN
                self.distance = abs(self.source[0] - self.target[0])
            elif (self.source[0] + self.source[1] ==
                    self.target[0] + self.target[1]):
                if self.source[0] > self.target[0]:
                    self.direction = Direction.RIGHT_DOWN
                else:
                    self.direction = Direction.LEFT_UP
                self.distance = abs(self.source[0] - self.target[0])
            elif self.source[1] > self.target[1]:
                if self.source[0] > self.target[0]:
                    self.direction = Direction.RIGHT_UP_UP
                else:
                    self.direction = Direction.LEFT_UP_UP
                self.distance = 1
            else:
                if self.source[0] > self.target[0]:
                    self.direction = Direction.RIGHT_DOWN_DOWN
                else:
                    self.direction = Direction.LEFT_DOWN_DOWN
                self.distance = 1

    def __str__(self):
        if self.action_type == ActionType.DROP:
            s = 'Action<Type: DROP, Piece: {}, Square: ({}, {})>'.format(
                self.piece, self.target[0] + 1, self.target[1] + 1
            )
        else:
            s = ('Action<Type: MOVE, Source: ({}, {}), '
                 'Target: ({}, {})>').format(
                self.source[0] + 1, self.source[1] + 1,
                self.target[0] + 1, self.target[1] + 1
            )
        return s

    @property
    def positive_source(self):
        return self.source

    @property
    def positive_target(self):
        return self.target

    @property
    def reversed_source(self):
        return 8 - self.source[0], 8 - self.source[1]

    @property
    def reversed_target(self):
        return 8 - self.target[0], 8 - self.target[1]


class Position(object):
    def __init__(self):
        self.board = shogi.Board()

        self.positive_board = Board(flip=False)
        self.reversed_board = Board(flip=True)

    def __str__(self):
        s = """step:{step}
{board}
--------------------------
{positive_board}
--------------------------
{reversed_board}
""".format(step=self.step_counter, board=self.board,
           positive_board=self.positive_board,
           reversed_board=self.reversed_board)

        return s

    def generate_action(self):
        return self.board.generate_legal_moves()

    def is_check(self):
        return self.board.is_check()

    def is_game_over(self):
        return self.board.is_game_over()

    def step(self, action):
        self.board.push(action)

        action = Action(action)
        self.positive_board.step(action)
        self.reversed_board.step(action)

    @property
    def turn(self):
        return self.board.turn

    @property
    def step_counter(self):
        return self.board.move_number

    def generate_turn_effect(self):
        """
        手番側の利きを計算する

        :return:
        """
        b = self.board
        if b.is_check():
            # 王から自分の駒に利きがあるか合法手のどちらか

            king_square = b.king_squares[b.turn]
            # 王の利きが自分の駒と重なっているもの
            moves = shogi.Board.attacks_from(
                piece_type=KING, square=b.king_squares[b.turn],
                occupied=b.occupied, move_color=b.turn
            ) & b.occupied[b.turn]
            to_square = bit_scan(moves)
            while to_square != -1 and to_square is not None:
                if not b.is_attacked_by(b.turn ^ 1, to_square):
                    # 相手の利きがない
                    delta = king_square - to_square
                    if not self.detect_wrong_evasion(move_color=b.turn,
                                                     king_square=king_square,
                                                     delta=delta):
                        yield to_square
                to_square = bit_scan(moves, to_square + 1)

            # 合法手からto_squareのみを抽出
            # 移動と駒を打つ場合では別なので、動かす場合のみを抜き出す
            moves = ((action.from_square, action.to_square)
                     for action in b.generate_legal_moves()
                     if action.from_square is not None)
            # 成り、成らずを一つにまとめる
            moves = set(moves)
            # 移動先のみを抽出
            to_squares = (to_square for _, to_square in moves)
            for to_square in to_squares:
                yield to_square
        else:
            # 王手でない場合
            for piece_type in PIECE_TYPES:
                movers = b.piece_bb[piece_type] & b.occupied[b.turn]
                from_square = bit_scan(movers)

                while from_square != -1 and from_square is not None:
                    if piece_type == KING:
                        moves = shogi.Board.attacks_from(
                            piece_type=piece_type, square=from_square,
                            occupied=b.occupied, move_color=b.turn
                        )
                    else:
                        pin_mask = self.detect_pin(square=from_square,
                                                   move_color=b.turn)

                        # ピンを考慮した移動可能なマス
                        moves = shogi.Board.attacks_from(
                            piece_type=piece_type, square=from_square,
                            occupied=b.occupied, move_color=b.turn
                        ) & pin_mask

                    to_square = bit_scan(moves)
                    while to_square != - 1 and to_square is not None:
                        if (piece_type != KING or
                                not b.is_attacked_by(b.turn ^ 1, to_square)):
                            # KING以外
                            # KINGでto_squareに相手の利きがない
                            yield to_square
                        to_square = bit_scan(moves, to_square + 1)
                    from_square = bit_scan(movers, from_square + 1)

    def generate_next_effect(self):
        """
        非手番側の駒の利きを求める

        :return:
        """
        # 王手はない

        b = self.board
        # 次の手番
        n = b.turn ^ 1
        for piece_type in PIECE_TYPES:
            movers = b.piece_bb[piece_type] & b.occupied[n]
            from_square = bit_scan(movers)

            while from_square != -1 and from_square is not None:
                if piece_type == KING:
                    moves = shogi.Board.attacks_from(
                        piece_type=piece_type, square=from_square,
                        occupied=b.occupied, move_color=n
                    )
                else:
                    pin_mask = self.detect_pin(square=from_square,
                                               move_color=n)
                    # ピンを考慮した移動可能なマス
                    moves = shogi.Board.attacks_from(
                        piece_type=piece_type, square=from_square,
                        occupied=b.occupied, move_color=n
                    ) & pin_mask

                to_square = bit_scan(moves)
                while to_square != - 1 and to_square is not None:
                    if (piece_type != KING or
                            not b.is_attacked_by(b.turn, to_square)):
                        # KING以外
                        # KINGでto_squareに相手の利きがない
                        yield to_square
                    to_square = bit_scan(moves, to_square + 1)
                from_square = bit_scan(movers, from_square + 1)

    def detect_pin(self, square, move_color):
        """
        move_colorのsquareにある駒がピンされているかを検出する
        :param square:
        :param move_color:
        :return:
        """
        # 手番側のoccupied bb
        turn_bb = self.board.occupied[move_color]
        # 非手番側のoccupied bb
        next_bb = self.board.occupied[move_color ^ 1]

        piece_bb = self.board.piece_bb
        king_bb = self.board.piece_bb[KING] & turn_bb

        cross_bb = (piece_bb[ROOK] | piece_bb[PROM_ROOK]) & next_bb
        # 縦方向
        index = (self.board.occupied.l90 >> (((square % 9) * 9) + 1)) & 127
        tmp = BB_FILE_ATTACKS[square][index]
        if (tmp & king_bb) > 0:
            if (tmp & cross_bb) > 0:
                return tmp
            elif (tmp & piece_bb[LANCE] & next_bb) > 0:
                # まだピンかわからない
                king_square = self.board.king_squares[move_color]
                flag1 = move_color == BLACK
                flag2 = king_square > square
                if flag1 == flag2:
                    return tmp
        # 横方向
        index = (self.board.occupied.bits >> (((square // 9) * 9) + 1)) & 127
        tmp = BB_RANK_ATTACKS[square][index]
        if (tmp & king_bb) > 0 and (tmp & cross_bb) > 0:
            return tmp

        diagonal_bb = (piece_bb[BISHOP] | piece_bb[PROM_BISHOP]) & next_bb
        # 斜め方向1
        index = (self.board.occupied.r45 >> BB_SHIFT_R45[square]) & 127
        tmp = BB_R45_ATTACKS[square][index]
        if (tmp & king_bb) > 0 and (tmp & diagonal_bb) > 0:
            return tmp
        # 斜め方向2
        index = (self.board.occupied.l45 >> BB_SHIFT_L45[square]) & 127
        tmp = BB_L45_ATTACKS[square][index]
        if (tmp & king_bb) > 0 and (tmp & diagonal_bb) > 0:
            return tmp

        return BB_ALL

    def detect_wrong_evasion(self, move_color, king_square, delta):
        """
        move_colorのkingが長い利きで王手されている場合に利きの上を遠ざかる方へは
        逃げられないので、それを検出する
        正確には遠ざかる方向への利きを検出する

        :param move_color:
        :param king_square:
        :param delta: king_square - 移動先の座標
        :return:
        """
        # 非手番側のoccupied bb
        next_bb = self.board.occupied[move_color ^ 1]

        piece_bb = self.board.piece_bb

        if abs(delta) == 10:
            # 斜め方向1
            diagonal_bb = (piece_bb[BISHOP] | piece_bb[PROM_BISHOP]) & next_bb

            index = (self.board.occupied.r45 >>
                     BB_SHIFT_R45[king_square]) & 127
            tmp = BB_R45_ATTACKS[king_square][index]

            return (tmp & diagonal_bb) > 0
        elif abs(delta) == 8:
            # 斜め方向2
            diagonal_bb = (piece_bb[BISHOP] | piece_bb[PROM_BISHOP]) & next_bb

            index = (self.board.occupied.l45 >>
                     BB_SHIFT_L45[king_square]) & 127
            tmp = BB_L45_ATTACKS[king_square][index]

            return (tmp & diagonal_bb) > 0
        elif abs(delta) == 9:
            # 縦方向
            if (delta == -9 and move_color == BLACK or
                    delta == 9 and move_color == WHITE):
                vertical_bb = (piece_bb[LANCE] | piece_bb[ROOK] |
                               piece_bb[PROM_ROOK]) & next_bb
            else:
                vertical_bb = (piece_bb[ROOK] | piece_bb[PROM_ROOK]) & next_bb
            index = (self.board.occupied.l90 >>
                     (((king_square % 9) * 9) + 1)) & 127
            tmp = BB_FILE_ATTACKS[king_square][index]

            return (tmp & vertical_bb) > 0
        else:
            # 横方向
            horizontal_bb = (piece_bb[ROOK] | piece_bb[PROM_ROOK]) & next_bb

            index = (self.board.occupied.bits >>
                     (((king_square // 9) * 9) + 1)) & 127
            tmp = BB_RANK_ATTACKS[king_square][index]

            return (tmp & horizontal_bb) > 0


def get_env():
    dotenv_path = Path(__file__).parents[0] / '.env'
    load_dotenv(str(dotenv_path))

    data_format = os.environ.get('DATA_FORMAT')
    use_cudnn = bool(os.environ.get('USE_CUDNN'))

    return data_format, use_cudnn


def run_random_action(seed, max_actions, n_episodes):
    np.random.seed(seed)

    data_format, use_cudnn = get_env()
    if data_format == 'NCHW':
        shape = (1, 1, 9, 9)
    else:
        shape = (1, 9, 9, 1)

    ph_board = tf.placeholder(shape=shape, dtype=tf.int32)
    ph_hand = tf.placeholder(shape=(1, 7), dtype=tf.int32)
    all_actions, black_count, white_count, black_check = AnnotationLayer(
        data_format=data_format, use_cudnn=use_cudnn
    )(ph_board, ph_hand)
    all_actions = [tf.squeeze(a) for a in all_actions]
    black_count = tf.squeeze(black_count)
    white_count = tf.squeeze(white_count)
    black_check = tf.squeeze(black_check)

    sess = tf.Session()

    cache_dir = 'random_state'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    v = -1
    for i in trange(max(0, v), n_episodes):
        if v == -1:
            name = os.path.join(cache_dir,
                                'seed{0}epoch{1:02d}.pickle'.format(seed, i))
            if not os.path.exists(name):
                random_state = np.random.get_state()
                joblib.dump(random_state, name)
        else:
            # 途中から再開
            i = v
            name = os.path.join(cache_dir,
                                'seed{0}epoch{1:02d}.pickle'.format(seed, v))
            random_state = joblib.load(name)
            np.random.set_state(random_state)
            # 通常ルートに戻るようにフラグを下ろす
            v = -1

        position = Position()

        while True:
            if position.turn == BLACK:
                feed_dict = {
                    ph_board: position.positive_board.board.reshape(shape),
                    ph_hand: position.positive_board.black_hand.reshape((1, 7))
                }
            else:
                feed_dict = {
                    ph_board: position.reversed_board.board.reshape(shape),
                    ph_hand: position.reversed_board.black_hand.reshape((1, 7))
                }
            actions, black_c, white_c, check = sess.run(
                [all_actions, black_count, white_count, black_check],
                feed_dict=feed_dict
            )

            msg = ""

            # 王手を確認
            if position.is_check() != check:
                tmp = 'ERROR: check is not match. {} v.s. {}\n'.format(
                    position.is_check(), check
                )
                tmp += str(position.board)
                tmp += '\nepisode: {}, {}\n'.format(i, position)

                msg += tmp

            # 行動が一致するか確認
            legal_actions = list(position.generate_action())
            converted_actions = [Action(a) for a in legal_actions]
            for action in converted_actions:
                index = get_action_index(action=action, turn=position.turn)
                a = actions[index]
                if position.turn == BLACK:
                    flag = a[action.target] != 1
                else:
                    flag = a[8 - action.target[0], 8 - action.target[1]] != 1
                if flag:
                    tmp = """ERROR: action is not match.
{action}
{action_board}
episode: {episode}, {position}
""".format(action=action, action_board=a, episode=i, position=position)
                    msg += tmp

            if len(legal_actions) == np.sum(actions):
                # 問題なし
                flag = False
            elif len(legal_actions) + 1 == np.sum(actions):
                # 打ち歩詰めの有無の違いは見逃す
                flag = not is_checkmate_with_dropping_fu(position=position,
                                                         actions=actions)
            else:
                # 何かがおかしい
                flag = True

            if flag:
                tmp = """ERROR: the number of actions is not match.
{expected} v.s. {actual}
episode: {episode}, {position}
""".format(expected=len(legal_actions), actual=np.sum(actions),
                    episode=i, position=position)

                msg += tmp

            # 手番側の利きを確認
            effect_count = np.zeros((9, 9), dtype=np.int32)
            if position.turn == BLACK:
                for e in position.generate_turn_effect():
                    q, r = divmod(e, 9)
                    effect_count[8 - r, q] += 1
            else:
                for e in position.generate_turn_effect():
                    q, r = divmod(e, 9)
                    effect_count[r, 8 - q] += 1
            if np.any(effect_count != black_c):
                tmp = """ERROR: turn effect is not match.
{expected}
-----------
{actual}
episode: {episode}, {position}
""".format(expected=effect_count, actual=black_c, episode=i, position=position)
                msg += tmp

            # 非手番側の利きを確認
            effect_count = np.zeros((9, 9), dtype=np.int32)
            if position.turn == BLACK:
                for e in position.generate_next_effect():
                    q, r = divmod(e, 9)
                    effect_count[8 - r, q] += 1
            else:
                for e in position.generate_next_effect():
                    q, r = divmod(e, 9)
                    effect_count[r, 8 - q] += 1
            if np.any(effect_count != white_c):
                tmp = """ERROR: next effect is not match.
{expected}
-----------
{actual}
episode: {episode}, {position}
""".format(expected=effect_count, actual=white_c, episode=i, position=position)
                msg += tmp

            if len(msg) != 0:
                print(msg)
                return

            if position.is_game_over():
                break

            action = np.random.choice(legal_actions)
            position.step(action=action)

            if position.step_counter >= max_actions:
                break

    sess.close()


def is_checkmate_with_dropping_fu(position, actions):
    if not position.board.has_piece_in_hand(PAWN, position.turn):
        return False

    king_square = position.board.king_squares[position.turn ^ 1]
    q, r = divmod(king_square, 9)
    if position.turn == BLACK:
        i, j = 8 - r, q
    else:
        i, j = r, 8 - q
    if j == 8:
        return False
    return actions[132][i, j + 1]


def get_action_index(action, turn):
    if action.action_type == ActionType.DROP:
        index = 132 + action.piece
    else:
        if turn == BLACK:
            if action.direction in (Direction.RIGHT_UP_UP,
                                    Direction.LEFT_UP_UP):
                index = (128 + action.direction - Direction.RIGHT_UP_UP +
                         action.promotion * 2)
            else:
                index = ((action.distance - 1) * 16 +
                         action.promotion * 8 + action.direction)
        else:
            if action.direction in (Direction.RIGHT_DOWN_DOWN,
                                    Direction.LEFT_DOWN_DOWN):
                index = (128 - action.direction + Direction.LEFT_DOWN_DOWN +
                         action.promotion * 2)
            else:
                index = ((action.distance - 1) * 16 + action.promotion * 8 +
                         7 - action.direction)
    return index


def main():
    run_random_action(seed=1, max_actions=512, n_episodes=100)


if __name__ == '__main__':
    main()
