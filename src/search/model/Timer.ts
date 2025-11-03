class Timer {
  private startTime = 0;
  private limit = 0;

  startSearch(ms: number) {
    this.startTime = performance.now();
    this.limit = ms;
  }

  isTimeUp(): boolean {
    return performance.now() - this.startTime >= this.limit;
  }

  getTimeElapsed(): number {
    return performance.now() - this.startTime;
  }
}

export const timer = new Timer();
