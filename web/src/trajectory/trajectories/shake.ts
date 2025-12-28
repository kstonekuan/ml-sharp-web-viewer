import { Vector3 } from "three";
import type { MaxOffset } from "../types";

/**
 * Create a left-right shake followed by an up-down shake trajectory.
 * Port of create_eye_trajectory_shake from camera.py lines 128-156
 */
export function createShakeTrajectory(
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	const numStepsTotal = numSteps * numRepeats;
	const numStepsHorizontal = Math.floor(numStepsTotal / 2);
	const numStepsVertical = numStepsTotal - numStepsHorizontal;

	const positions: Vector3[] = [];

	// Horizontal shake
	for (let i = 0; i < numStepsHorizontal; i++) {
		const t = (i / numStepsHorizontal) * numRepeats;
		const x = offset.x * Math.sin(2 * Math.PI * t);
		positions.push(new Vector3(x, 0, distanceMeters));
	}

	// Vertical shake
	for (let i = 0; i < numStepsVertical; i++) {
		const t = (i / numStepsVertical) * numRepeats;
		const y = offset.y * Math.sin(2 * Math.PI * t);
		positions.push(new Vector3(0, y, distanceMeters));
	}

	return positions;
}
