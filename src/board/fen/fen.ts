import {Board} from "../board";
import {BOARD_SIZE} from "../constants";
import {clearBoard} from "../board";
import {getSquare} from "../utils/utils";
import {Piece, Side} from "../types";
import {setPiece, flipSide} from "../board";

const isValidFENChar = (char: string) => {
  return 'kad0123456789'.includes(char)
}

const charToPiece: Record<string, Piece> = {
  'k': Piece.KING,
  'a': Piece.ATTACKER,
  'd': Piece.DEFENDER,
}

const charToSide: Record<string, number> = {
  'a': Side.ATTACKERS,
  'd': Side.DEFENDERS,
}

const pieceToChar = {
  [Piece.KING]: 'k',
  [Piece.ATTACKER]: 'a',
  [Piece.DEFENDER]: 'd',
}

const sideToChar: Record<Side, string> = {
  [Side.ATTACKERS]: 'a',
  [Side.DEFENDERS]: 'd',
}

const isNumberChar = (char: string) => {
  return '0123456789'.includes(char)
}

const isSideValid = (char: string) => {
  return 'ad'.includes(char)
}

const setRowFromFEN = (rowFEN: string, rowIndex: number, board: Board) => {
  let colIndex = 0
  let numberCache = ''

  for (let i = 0; i < rowFEN.length; i++) {
    const char = rowFEN[i]

    if (!isValidFENChar(char)) {
      throw new Error(`Invalid FEN character: ${char}`)
    }

    if (isNumberChar(char)) {
      numberCache += char
      continue
    }

    if (numberCache.length > 0) {
      const number = parseInt(numberCache, 10)
      colIndex += number
      numberCache = ''
    }

    // Set piece on board
    const sq = getSquare(rowIndex, colIndex)
    console.log({
      piece: charToPiece[char]
    })
    setPiece(board, sq, charToPiece[char])
    colIndex++
  }
}

export const setFEN = (board: Board, fen: string) => {
  clearBoard(board)
  const [rows, side] = fen.split(' ')

  const splittedRows = rows.split('/')

  if (splittedRows.length !== BOARD_SIZE) {
    throw new Error(`Invalid FEN rows: ${rows}`)
  }

  splittedRows
    .forEach((rowFEN, rowIndex) => {
      setRowFromFEN(rowFEN, BOARD_SIZE - 1 - rowIndex, board)
    })

  console.log({splittedRows, side})

  const sideToSet = charToSide[side]
  if (board.sideToMove !== sideToSet) {
    flipSide(board)
  }
}

export const getFEN = (board: Board) => {
  let fenRows = []

  for (let row = BOARD_SIZE - 1; row >= 0; row--) {
    let emptyCount = 0
    let fenRow = ''

    for (let col = 0; col < BOARD_SIZE; col++) {
      const sq = getSquare(row, col)
      const piece = board.board[sq] as Piece

      if (piece === Piece.EMPTY) {
        emptyCount++
        continue
      }

      if (emptyCount > 0) {
        fenRow += emptyCount.toString()
        emptyCount = 0
      }

      fenRow += pieceToChar[piece]
    }

    if (emptyCount > 0) {
      fenRow += emptyCount.toString()
    }

    fenRows.push(fenRow)
  }

  return fenRows.join('/') + ' ' + sideToChar[board.sideToMove as Side]
}
