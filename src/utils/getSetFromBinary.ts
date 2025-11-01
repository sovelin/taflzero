export const getSetFromBinary = (array: Uint8Array): Set<number> => {
  const resultSet = new Set<number>();
  for (let i = 0; i < array.length; i++) {
    if (array[i]) {
      resultSet.add(i);
    }
  }
  return resultSet;
}
