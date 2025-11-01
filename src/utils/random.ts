let rngState = 0x9e3779b97f4a7c15n; // фиксированный сид

export function random64() {
  let x = rngState;
  x ^= x >> 12n;
  x ^= x << 25n;
  x ^= x >> 27n;
  rngState = x;
  return x * 0x2545F4914F6CDD1Dn & 0xFFFFFFFFFFFFFFFFn;
}
