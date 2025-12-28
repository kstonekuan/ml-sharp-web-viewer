import { Vector3 } from "three";
import type { MaxOffset } from "../types";

/**
 * Create a rotating trajectory around the scene.
 * Port of create_eye_trajectory_rotate from camera.py lines 159-180
 */
export function createRotateTrajectory(
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	const numStepsTotal = numSteps * numRepeats;
	const positions: Vector3[] = [];

	for (let i = 0; i < numStepsTotal; i++) {
		const t = (i / numStepsTotal) * numRepeats;
		const x = offset.x * Math.sin(2 * Math.PI * t);
		const y = offset.y * Math.cos(2 * Math.PI * t);
		positions.push(new Vector3(x, y, distanceMeters));
	}

	return positions;
}
