export type Vec3 = [number, number, number];

export function centroid(points: Vec3[]): Vec3 {
	const c: Vec3 = [0, 0, 0];
	for (const p of points) for (let i = 0; i < 3; i++) c[i] += p[i] / points.length;
	return c;
}

/**
 * Rotation that best maps the centred `mobile` points onto the centred `target` points
 * (Horn's quaternion method), and the RMSD after applying it. Both lists must pair up one to one.
 */
export function superpose(mobile: Vec3[], target: Vec3[]): { rotation: number[][]; rmsd: number } {
	const cm = centroid(mobile);
	const ct = centroid(target);
	const S = [
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0]
	];
	for (let n = 0; n < mobile.length; n++) {
		for (let a = 0; a < 3; a++) {
			for (let b = 0; b < 3; b++) S[a][b] += (mobile[n][a] - cm[a]) * (target[n][b] - ct[b]);
		}
	}
	const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
	const N = [
		[xx + yy + zz, yz - zy, zx - xz, xy - yx],
		[yz - zy, xx - yy - zz, xy + yx, zx + xz],
		[zx - xz, xy + yx, -xx + yy - zz, yz + zy],
		[xy - yx, zx + xz, yz + zy, -xx - yy + zz]
	];

	// Power iteration on N shifted to be positive definite finds its largest eigenvector, the best rotation.
	const shift = N.flat().reduce((sum, v) => sum + Math.abs(v), 0);
	let q = [1, 0, 0, 0];
	for (let iter = 0; iter < 500; iter++) {
		const next = N.map((row, i) => row.reduce((sum, v, j) => sum + v * q[j], 0) + shift * q[i]);
		const norm = Math.hypot(...next);
		q = next.map((v) => v / norm);
	}
	const [w, x, y, z] = q;
	const rotation = [
		[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
		[2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
		[2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]
	];

	let sq = 0;
	for (let n = 0; n < mobile.length; n++) {
		const p = mobile[n].map((v, i) => v - cm[i]);
		for (let a = 0; a < 3; a++) {
			const r = rotation[a][0] * p[0] + rotation[a][1] * p[1] + rotation[a][2] * p[2];
			sq += (r - (target[n][a] - ct[a])) ** 2;
		}
	}
	return { rotation, rmsd: Math.sqrt(sq / mobile.length) };
}
