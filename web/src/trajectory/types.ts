import type * as THREE from "three";

export type TrajectoryType =
	| "swipe"
	| "shake"
	| "rotate"
	| "rotate_forward"
	| "forward";

export interface TrajectoryParams {
	type: TrajectoryType;
	maxDisparity: number;
	maxZoom: number;
	distanceMeters: number;
	numSteps: number;
	numRepeats: number;
}

export const DEFAULT_TRAJECTORY_PARAMS: TrajectoryParams = {
	type: "rotate_forward",
	maxDisparity: 0.08,
	maxZoom: 0.15,
	distanceMeters: 0.0,
	numSteps: 60,
	numRepeats: 1,
};

export interface MaxOffset {
	x: number;
	y: number;
	z: number;
}

export type PlayerState = "stopped" | "playing" | "paused";

export type TrajectoryGenerator = (
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
) => THREE.Vector3[];
