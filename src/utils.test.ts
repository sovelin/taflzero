import {describe, it, expect} from "vitest";
import {getBinary} from "./utils";

describe('Utility Tests', () => {
  describe('getBinary', () => {
    it('should convert number to inverted binary string', () => {
      const value = 0b10111011;
      const binaryString = getBinary(value, 8);
      expect(binaryString).toBe('11011101');
    })
  })
})
