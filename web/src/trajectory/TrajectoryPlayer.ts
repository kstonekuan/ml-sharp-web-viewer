import type { Vector3 } from "three";
import type { PlayerState } from "./types";

/**
 * Manages playback state for camera trajectory animation.
 */
export class TrajectoryPlayer {
	private positions: Vector3[] = [];
	private currentFrame = 0;
	private state: PlayerState = "stopped";
	private lastFrameTime = 0;
	private frameInterval: number;

	public onComplete?: () => void;
	public onFrameChange?: (frame: number, total: number) => void;
	public onStateChange?: (state: PlayerState) => void;

	constructor(fps = 30) {
		this.frameInterval = 1000 / fps;
	}

	setTrajectory(positions: Vector3[]): void {
		this.positions = positions;
		this.currentFrame = 0;
		this.state = "stopped";
		this.onStateChange?.("stopped");
		this.onFrameChange?.(0, positions.length);
	}

	play(): void {
		if (this.positions.length === 0) return;
		this.state = "playing";
		this.lastFrameTime = performance.now();
		this.onStateChange?.("playing");
	}

	pause(): void {
		if (this.state !== "playing") return;
		this.state = "paused";
		this.onStateChange?.("paused");
	}

	stop(): void {
		this.state = "stopped";
		this.currentFrame = 0;
		this.onStateChange?.("stopped");
		this.onFrameChange?.(0, this.positions.length);
	}

	reset(): void {
		this.currentFrame = 0;
		this.onFrameChange?.(0, this.positions.length);
		if (this.state === "playing") {
			this.lastFrameTime = performance.now();
		}
	}

	/**
	 * Update the player state and return the current eye position.
	 * Should be called every frame in the render loop.
	 *
	 * @param currentTime - Current timestamp from requestAnimationFrame
	 * @returns Current eye position, or null if not playing/no trajectory
	 */
	update(currentTime: number): Vector3 | null {
		if (this.state !== "playing" || this.positions.length === 0) {
			return this.getCurrentPosition();
		}

		const elapsed = currentTime - this.lastFrameTime;

		if (elapsed >= this.frameInterval) {
			this.currentFrame++;
			this.lastFrameTime = currentTime;

			if (this.currentFrame >= this.positions.length) {
				this.currentFrame = 0;
				this.state = "stopped";
				this.onStateChange?.("stopped");
				this.onComplete?.();
			}

			this.onFrameChange?.(this.currentFrame, this.positions.length);
		}

		return this.getCurrentPosition();
	}

	getCurrentPosition(): Vector3 | null {
		if (this.positions.length === 0) return null;
		return this.positions[this.currentFrame] ?? null;
	}

	getCurrentFrame(): number {
		return this.currentFrame;
	}

	getTotalFrames(): number {
		return this.positions.length;
	}

	getState(): PlayerState {
		return this.state;
	}

	isPlaying(): boolean {
		return this.state === "playing";
	}

	setFrame(frame: number): void {
		if (frame >= 0 && frame < this.positions.length) {
			this.currentFrame = frame;
			this.onFrameChange?.(frame, this.positions.length);
		}
	}
}
