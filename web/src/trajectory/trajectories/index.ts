import type { Vector3 } from "three";
import type { MaxOffset, TrajectoryType } from "../types";
import { createForwardTrajectory } from "./forward";
import { createRotateTrajectory } from "./rotate";
import { createRotateForwardTrajectory } from "./rotateForward";
import { createShakeTrajectory } from "./shake";
import { createSwipeTrajectory } from "./swipe";

export { createForwardTrajectory } from "./forward";
export { createRotateTrajectory } from "./rotate";
export { createRotateForwardTrajectory } from "./rotateForward";
export { createShakeTrajectory } from "./shake";
export { createSwipeTrajectory } from "./swipe";

/**
 * Create eye trajectory for the given trajectory type.
 * Port of create_eye_trajectory from camera.py lines 74-110
 */
export function createEyeTrajectory(
	type: TrajectoryType,
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	switch (type) {
		case "swipe":
			return createSwipeTrajectory(
				offset,
				distanceMeters,
				numSteps,
				numRepeats,
			);
		case "shake":
			return createShakeTrajectory(
				offset,
				distanceMeters,
				numSteps,
				numRepeats,
			);
		case "rotate":
			return createRotateTrajectory(
				offset,
				distanceMeters,
				numSteps,
				numRepeats,
			);
		case "rotate_forward":
			return createRotateForwardTrajectory(
				offset,
				distanceMeters,
				numSteps,
				numRepeats,
			);
		case "forward":
			return createForwardTrajectory(
				offset,
				distanceMeters,
				numSteps,
				numRepeats,
			);
		default:
			throw new Error(`Invalid trajectory type: ${type}`);
	}
}
