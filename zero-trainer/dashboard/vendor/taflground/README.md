# TaflGround

Vue 3 Tafl board UI component.

`taflground` is now fully controlled:
- it does not store or compute game rules;
- it does not apply moves internally;
- it renders position only from `piecesList`;
- it emits `move` on user interaction, and the outer app decides validity and next state.

## Install

```bash
npm i taflground vue zevratafl
```

Or with `yarn`:

```bash
yarn add taflground vue zevratafl
```

## Usage

As a plugin:

```ts
import {createApp} from "vue";
import App from "./App.vue";
import TaflGround from "taflground";

createApp(App).use(TaflGround).mount("#app");
```

Or with named imports:

```ts
import {TaflBoard, BoardTheme, PieceTheme} from "taflground";
```

## Controlled Example

```vue
<script setup lang="ts">
import {computed, ref} from "vue";
import {TaflBoard, BoardTheme, PieceTheme} from "taflground";
import type {Move, Piece} from "taflground";

const pieces = ref<Piece[]>([
  {sq: 15, type: "king"},
  {sq: 2, type: "attacker"},
  {sq: 4, type: "defender"},
]);

const selectableSquares = ref(new Set<number>([15, 2]));

const possibleMoves = computed<Move[]>(() => [
  {from: 15, to: 14},
  {from: 15, to: 16},
]);

const applyMove = (move: Move) => {
  // Validation and next position calculation must live outside the board
  // (engine/store/server).
  pieces.value = pieces.value.map((piece) =>
    piece.sq === move.from ? {...piece, sq: move.to} : piece
  );
};
</script>

<template>
  <TaflBoard
    :pieces-list="pieces"
    :possible-moves="possibleMoves"
    :selectable-squares="selectableSquares"
    :size="800"
    :board-theme="BoardTheme.CLASSIC_BLUE"
    :piece-theme="PieceTheme.ALPHA"
    :highlight-move="null"
    @move="applyMove"
  />
</template>
```

## TaflBoard Props

- `piecesList: Piece[]` - current board pieces state.
- `selectableSquares: Set<number>` - squares that can be selected as move source.
- `possibleMoves: Move[]` - moves to highlight for currently selected piece.
- `highlightMove: Move | null` - externally controlled move highlight.
- `size: number` - board size in pixels.
- `boardTheme: BoardThemeValue` - board theme.
- `pieceTheme: PieceThemeValue` - piece theme.

## Events

- `move` - user attempted a move (`{ from: number, to: number }`).

The component does not mutate `piecesList` automatically. To display a move, update `piecesList` from outside.
