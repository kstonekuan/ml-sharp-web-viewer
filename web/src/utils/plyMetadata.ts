import { Matrix3, Matrix4 } from "three";

export type ColorSpace = "sRGB" | "linearRGB";

export interface PlyMetadata {
	extrinsics: Matrix4;
	intrinsics: Matrix3;
	imageSize: [number, number];
	focalLength: number;
	colorSpace: ColorSpace;
	hasMetadata: boolean;
}

const DEFAULT_METADATA: PlyMetadata = {
	extrinsics: new Matrix4(),
	intrinsics: new Matrix3(),
	imageSize: [640, 480],
	focalLength: 0, // 0 means "compute from image size"
	colorSpace: "sRGB",
	hasMetadata: false,
};

/**
 * Estimate focal length from image size for ~50° vertical FOV (typical phone camera).
 * focalLength = height / (2 * tan(fov/2))
 * For 50° FOV: focalLength ≈ height / 0.93 ≈ height * 1.07
 */
export function estimateFocalLength(imageSize: [number, number]): number {
	const height = imageSize[1];
	// Approximate focal length for ~50° vertical FOV
	return height * 1.07;
}

/** Property type info with byte size */
interface PropertyInfo {
	name: string;
	type: string;
	byteSize: number;
}

/** Element info with properties and total byte size */
interface ElementInfo {
	name: string;
	count: number;
	properties: PropertyInfo[];
	bytesPerElement: number;
}

/** Get byte size for PLY property type */
function getPropertyByteSize(type: string): number {
	switch (type) {
		case "char":
		case "uchar":
		case "int8":
		case "uint8":
			return 1;
		case "short":
		case "ushort":
		case "int16":
		case "uint16":
			return 2;
		case "int":
		case "uint":
		case "int32":
		case "uint32":
		case "float":
		case "float32":
			return 4;
		case "double":
		case "float64":
			return 8;
		default:
			return 4; // Default to float
	}
}

/**
 * Parse PLY header to find element declarations with property types.
 */
function parsePlyHeader(text: string): {
	headerEndIndex: number;
	elements: ElementInfo[];
	format: "binary_little_endian" | "binary_big_endian" | "ascii";
} {
	const lines = text.split("\n");
	const elements: ElementInfo[] = [];
	let currentElement: ElementInfo | null = null;
	let headerEndIndex = 0;
	let byteOffset = 0;
	let format: "binary_little_endian" | "binary_big_endian" | "ascii" = "ascii";

	for (const line of lines) {
		byteOffset += line.length + 1; // +1 for newline

		const trimmed = line.trim();
		if (trimmed === "end_header") {
			headerEndIndex = byteOffset;
			break;
		}

		if (trimmed.startsWith("format ")) {
			const parts = trimmed.split(/\s+/);
			if (parts[1] === "binary_little_endian") {
				format = "binary_little_endian";
			} else if (parts[1] === "binary_big_endian") {
				format = "binary_big_endian";
			}
		} else if (trimmed.startsWith("element ")) {
			// Save previous element
			if (currentElement) {
				currentElement.bytesPerElement = currentElement.properties.reduce(
					(sum, p) => sum + p.byteSize,
					0,
				);
				elements.push(currentElement);
			}

			const parts = trimmed.split(/\s+/);
			if (parts.length >= 3 && parts[1]) {
				currentElement = {
					name: parts[1],
					count: Number.parseInt(parts[2] ?? "0", 10),
					properties: [],
					bytesPerElement: 0,
				};
			}
		} else if (trimmed.startsWith("property ") && currentElement) {
			const parts = trimmed.split(/\s+/);
			// property <type> <name>
			if (parts.length >= 3 && parts[1] && parts[2]) {
				const type = parts[1];
				const name = parts[parts.length - 1] ?? "";
				currentElement.properties.push({
					name,
					type,
					byteSize: getPropertyByteSize(type),
				});
			}
		}
	}

	// Save last element
	if (currentElement) {
		currentElement.bytesPerElement = currentElement.properties.reduce(
			(sum, p) => sum + p.byteSize,
			0,
		);
		elements.push(currentElement);
	}

	return { headerEndIndex, elements, format };
}

/**
 * Read floats from DataView at offset
 */
function readFloats(
	view: DataView,
	offset: number,
	count: number,
	littleEndian: boolean,
): number[] {
	const result: number[] = [];
	for (let i = 0; i < count; i++) {
		result.push(view.getFloat32(offset + i * 4, littleEndian));
	}
	return result;
}

/**
 * Read uints from DataView at offset
 */
function readUints(
	view: DataView,
	offset: number,
	count: number,
	littleEndian: boolean,
): number[] {
	const result: number[] = [];
	for (let i = 0; i < count; i++) {
		result.push(view.getUint32(offset + i * 4, littleEndian));
	}
	return result;
}

/**
 * Try to extract metadata from PLY file.
 * The SHARP PLY format stores metadata in supplementary elements.
 *
 * @param buffer - The ArrayBuffer of the PLY file
 * @returns Parsed metadata or defaults
 */
export function parsePlyMetadata(buffer: ArrayBuffer): PlyMetadata {
	try {
		// Read header as text
		const decoder = new TextDecoder("utf-8");
		const headerBytes = new Uint8Array(buffer.slice(0, 10000));
		const headerText = decoder.decode(headerBytes);

		const { headerEndIndex, elements, format } = parsePlyHeader(headerText);

		// Only support binary little endian (most common)
		if (format !== "binary_little_endian") {
			console.log("[plyMetadata] Format not binary_little_endian:", format);
			return { ...DEFAULT_METADATA };
		}

		// Find supplement elements
		const elementMap = new Map<string, ElementInfo>();
		for (const el of elements) {
			elementMap.set(el.name, el);
		}

		const hasIntrinsic =
			elementMap.has("intrinsic") || elementMap.has("supplement");
		const hasImageSize = elementMap.has("image_size");

		if (!hasIntrinsic && !hasImageSize) {
			console.log("[plyMetadata] No SHARP metadata elements found");
			return { ...DEFAULT_METADATA };
		}

		// Calculate byte offset to each element
		const view = new DataView(buffer);
		let offset = headerEndIndex;
		const elementOffsets = new Map<string, number>();

		for (const el of elements) {
			elementOffsets.set(el.name, offset);
			offset += el.count * el.bytesPerElement;
		}

		const metadata: PlyMetadata = {
			...DEFAULT_METADATA,
			hasMetadata: true,
		};

		const littleEndian = true;

		// Read image_size if present
		const imageSizeEl = elementMap.get("image_size");
		if (imageSizeEl && imageSizeEl.count >= 1) {
			const imageSizeOffset = elementOffsets.get("image_size");
			if (imageSizeOffset !== undefined) {
				// image_size has 2 uint properties: width, height
				const values = readUints(view, imageSizeOffset, 2, littleEndian);
				if (values[0] !== undefined && values[1] !== undefined) {
					metadata.imageSize = [values[0], values[1]];
					console.log("[plyMetadata] Image size:", metadata.imageSize);
				}
			}
		}

		// Read intrinsic if present
		const intrinsicEl = elementMap.get("intrinsic");
		if (intrinsicEl && intrinsicEl.count >= 1) {
			const intrinsicOffset = elementOffsets.get("intrinsic");
			if (intrinsicOffset !== undefined) {
				const propCount = intrinsicEl.properties.length;

				if (propCount === 9) {
					// 3x3 intrinsic matrix
					const values = readFloats(view, intrinsicOffset, 9, littleEndian);
					// Focal length is at [0,0] and [1,1]
					const fx = values[0] ?? 512;
					const fy = values[4] ?? 512;
					metadata.focalLength = (fx + fy) / 2;
					console.log(
						"[plyMetadata] Focal length (from 3x3):",
						metadata.focalLength,
					);

					// Set intrinsics matrix
					metadata.intrinsics.set(
						values[0] ?? 1,
						values[1] ?? 0,
						values[2] ?? 0,
						values[3] ?? 0,
						values[4] ?? 1,
						values[5] ?? 0,
						values[6] ?? 0,
						values[7] ?? 0,
						values[8] ?? 1,
					);
				} else if (propCount === 4) {
					// Legacy: fx, fy, width, height
					const values = readFloats(view, intrinsicOffset, 4, littleEndian);
					const fx = values[0] ?? 512;
					const fy = values[1] ?? 512;
					metadata.focalLength = (fx + fy) / 2;
					console.log(
						"[plyMetadata] Focal length (legacy):",
						metadata.focalLength,
					);

					// If image_size wasn't found, use from intrinsics
					if (
						!imageSizeEl &&
						values[2] !== undefined &&
						values[3] !== undefined
					) {
						metadata.imageSize = [Math.round(values[2]), Math.round(values[3])];
						console.log(
							"[plyMetadata] Image size (from intrinsic):",
							metadata.imageSize,
						);
					}
				}
			}
		}

		// Read extrinsic if present
		const extrinsicEl = elementMap.get("extrinsic");
		if (extrinsicEl && extrinsicEl.count >= 1) {
			const extrinsicOffset = elementOffsets.get("extrinsic");
			if (extrinsicOffset !== undefined) {
				const propCount = extrinsicEl.properties.length;

				if (propCount === 16) {
					// 4x4 extrinsic matrix
					const values = readFloats(view, extrinsicOffset, 16, littleEndian);
					// Three.js Matrix4 is column-major, PLY stores row-major
					metadata.extrinsics.set(
						values[0] ?? 1,
						values[4] ?? 0,
						values[8] ?? 0,
						values[12] ?? 0,
						values[1] ?? 0,
						values[5] ?? 1,
						values[9] ?? 0,
						values[13] ?? 0,
						values[2] ?? 0,
						values[6] ?? 0,
						values[10] ?? 1,
						values[14] ?? 0,
						values[3] ?? 0,
						values[7] ?? 0,
						values[11] ?? 0,
						values[15] ?? 1,
					);
					console.log("[plyMetadata] Extrinsics loaded");
				} else if (propCount === 12) {
					// Legacy: 3x4 matrix
					const values = readFloats(view, extrinsicOffset, 12, littleEndian);
					metadata.extrinsics.set(
						values[0] ?? 1,
						values[4] ?? 0,
						values[8] ?? 0,
						0,
						values[1] ?? 0,
						values[5] ?? 1,
						values[9] ?? 0,
						0,
						values[2] ?? 0,
						values[6] ?? 0,
						values[10] ?? 1,
						0,
						values[3] ?? 0,
						values[7] ?? 0,
						values[11] ?? 0,
						1,
					);
					console.log("[plyMetadata] Extrinsics loaded (legacy 3x4)");
				}
			}
		}

		// Read color_space if present
		const colorSpaceEl = elementMap.get("color_space");
		if (colorSpaceEl && colorSpaceEl.count >= 1) {
			const colorSpaceOffset = elementOffsets.get("color_space");
			if (colorSpaceOffset !== undefined) {
				const value = view.getUint8(colorSpaceOffset);
				metadata.colorSpace = decodeColorSpace(value);
				console.log("[plyMetadata] Color space:", metadata.colorSpace);
			}
		}

		// Also check for supplement element (alternative format)
		const supplementEl = elementMap.get("supplement");
		if (supplementEl && !intrinsicEl) {
			// Some PLY files store all metadata in a single supplement element
			// Try to parse if it has the expected properties
			console.log(
				"[plyMetadata] Found supplement element with",
				supplementEl.properties.length,
				"properties",
			);
		}

		return metadata;
	} catch (error) {
		console.error("[plyMetadata] Error parsing:", error);
		return { ...DEFAULT_METADATA };
	}
}

/**
 * Decode color space from byte value.
 * Matches decode_color_space in color_space.py
 */
export function decodeColorSpace(value: number): ColorSpace {
	return value === 0 ? "linearRGB" : "sRGB";
}
