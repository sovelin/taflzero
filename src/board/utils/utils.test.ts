import {
  getSquareFromAlgebraic,
  getLeftNeighbor,
  getRightNeighbor,
  getTopNeighbor,
  getBottomNeighbor,
  getTopLeftNeighbor, getTopRightNeighbor, getBottomLeftNeighbor, getVerticalHorizontalNeighbors, getAllNeighbors
} from "./utils";
import {describe, expect, it} from "vitest";

describe('Utility Functions Tests', () => {
  describe('siblings', () => {
    it('left sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const leftSibling = getLeftNeighbor(sq);
      expect(leftSibling).toBe(getSquareFromAlgebraic("c4"));
    })

    it('left sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("a1");
      const leftSibling = getLeftNeighbor(sq);
      expect(leftSibling).toBeNull();
    })

    it('right sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const rightSibling = getRightNeighbor(sq);
      expect(rightSibling).toBe(getSquareFromAlgebraic("e4"));
    })

    it('right sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("k8");
      const rightSibling = getRightNeighbor(sq);
      expect(rightSibling).toBeNull();
    })

    it('top sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const topSibling = getTopNeighbor(sq);
      expect(topSibling).toBe(getSquareFromAlgebraic("d5"));
    })

    it('top sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("e11");
      const topSibling = getTopNeighbor(sq);
      expect(topSibling).toBeNull();
    })

    it('bottom sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const bottomSibling = getTopNeighbor(sq);
      expect(bottomSibling).toBe(getSquareFromAlgebraic("d5"));
    })

    it('bottom sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("e1");
      const bottomSibling = getBottomNeighbor(sq);
      expect(bottomSibling).toBeNull();
    })

    it('top left sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const topLeftSibling = getTopLeftNeighbor(sq);
      expect(topLeftSibling).toBe(getSquareFromAlgebraic("c5"));
    })

    it('top left sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("a8");
      const topLeftSibling = getTopLeftNeighbor(sq);
      expect(topLeftSibling).toBeNull();
    })

    it('top right sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const topRightSibling = getTopRightNeighbor(sq);
      expect(topRightSibling).toBe(getSquareFromAlgebraic("e5"));
    })

    it('top right sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("k8");
      const topRightSibling = getTopRightNeighbor(sq);
      expect(topRightSibling).toBeNull();
    })

    it('bottom left sibling exists', () => {
      const sq = getSquareFromAlgebraic("d4");
      const bottomLeftSibling = getBottomLeftNeighbor(sq);
      expect(bottomLeftSibling).toBe(getSquareFromAlgebraic("c3"));
    })

    it('bottom left sibling does not exist', () => {
      const sq = getSquareFromAlgebraic("a1");
      const bottomLeftSibling = getTopLeftNeighbor(sq);
      expect(bottomLeftSibling).toBeNull();
    })

    describe('mass sibling retrieval', () => {
      const expectExactlySiblings = (expectedAlgebraic: string[], sqAlgebraic: number[]) => {
        expect(sqAlgebraic.length).toBe(expectedAlgebraic.length);

        for (const alg of expectedAlgebraic) {
          const sq = getSquareFromAlgebraic(alg);
          expect(sqAlgebraic).toContain(sq);
        }
      }

      describe('vertical/horizontal siblings', () => {
        it('center square', () => {
          const siblings = getVerticalHorizontalNeighbors(getSquareFromAlgebraic("e6"))

          expectExactlySiblings(
            ["e7", "e5", "d6", "f6"],
            siblings
          )
        })

        it('edge square', () => {
          const siblings = getVerticalHorizontalNeighbors(getSquareFromAlgebraic("a6"))

          expectExactlySiblings(
            ["a7", "a5", "b6"],
            siblings
          )
        })

        it('corner square', () => {
          const siblings = getVerticalHorizontalNeighbors(getSquareFromAlgebraic("a1"))

          expectExactlySiblings(
            ["a2", "b1"],
            siblings
          )
        })
      })

      describe('all siblings', () => {
        it('center square', () => {
          const siblings = getAllNeighbors(getSquareFromAlgebraic("e6"))

          expectExactlySiblings(
            ["e7", "e5", "d6", "f6", "d7", "f7", "d5", "f5"],
            siblings
          )
        })

        it('edge square', () => {
          const siblings = getAllNeighbors(getSquareFromAlgebraic("a6"))

          expectExactlySiblings(
            ["a7", "a5", "b6", "b7", "b5"],
            siblings
          )
        })

        it('corner square', () => {
          const siblings = getAllNeighbors(getSquareFromAlgebraic("a1"))

          expectExactlySiblings(
            ["a2", "b1", "b2"],
            siblings
          )
        })
      })
    })
  })
})
