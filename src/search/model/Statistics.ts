export class Statistics {
  nodes = 0;

  reset() {
    this.nodes = 0;
  }

  incrementNodes(count = 1) {
    this.nodes += count;
  }
}

export const statistics = new Statistics();
