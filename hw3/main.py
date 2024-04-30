import numpy as np
import argparse
import pickle
from PIL import Image
import os
from tqdm import tqdm

def power_iteration(A, iter_count):
    """Performs power iteration to find the dominant eigenvalue and eigenvector of matrix A."""
    randoms = np.random.rand(A.shape[1])
    for _ in tqdm(range(iter_count)):
        trans = np.dot(A, randoms)
        trans_norm = np.linalg.norm(trans)
        randoms = trans / trans_norm

    val = np.dot(randoms.T, np.dot(A, randoms))
    vec = randoms
    return val, vec

def simple_svd(A, iter_count=10, nval=None):
    """Performs SVD using the power iteration method."""
    m, n = A.shape
    if nval is None:
        nval = min(m, n)
    AAT = A @ A.T
    val = []
    vec = []

    for _ in tqdm(range(nval)):
        va, ve = power_iteration(AAT, iter_count)
        val.append(va)
        vec.append(ve)
        AAT -= va * np.outer(ve, ve)

    S = np.sqrt(val)
    U = np.vstack(vec).T
    V = (A.T @ U) @ np.diag(1 / S)
    return U, S, V.T

def advanced_power_iteration(A, iter_count, epsilon=1e-10):
    """Computes the orthogonal matrix for A using QR iterations."""
    n, _ = A.shape
    ort = np.eye(n)

    for _ in tqdm(range(iter_count)):
        z = A @ ort
        ort, tri = np.linalg.qr(z)
        diff = np.sum(np.abs(np.diag(tri))) / np.sum(np.abs(tri))

        if diff < epsilon:
            break

    return ort

def advanced_svd(A, iter_count=10, nval=None):
    """Computes the SVD using advanced QR iterations."""
    m, n = A.shape
    if nval is None or nval > min(m, n):
        nval = min(m, n)

    V = advanced_power_iteration(A.T @ A, iter_count)[:, :nval]
    S = np.sqrt(np.maximum(np.diag(V.T @ A.T @ A @ V), 0))
    U = A @ V @ np.linalg.inv(np.diag(S))

    return U, S, V.T


def compress_image_svd(image, method, compression):
    """Compresses an image using SVD by breaking it down into its color components."""
    Rc, Gc, Bc = image[:,:,0].astype(np.float32), image[:,:,1].astype(np.float32), image[:,:,2].astype(np.float32)
    methods = {
        'numpy': lambda x: np.linalg.svd(x, full_matrices=False),
        'simple': lambda x: simple_svd(x, iter_count=10, nval=min(x.shape)//compression),
        'advanced': lambda x: advanced_svd(x, iter_count=10, nval=min(x.shape)//compression)
    }
    Ur, Sr, Vr = methods[method](Rc)
    Ug, Sg, Vg = methods[method](Gc)
    Ub, Sb, Vb = methods[method](Bc)

    k = min(len(Sr), len(Sg), len(Sb)) // compression
    compressed_data = {'Ur': Ur[:, :k], 'Sr': Sr[:k], 'Vr': Vr[:k, :], 'Ug': Ug[:, :k], 'Sg': Sg[:k], 'Vg': Vg[:k, :], 'Ub': Ub[:, :k], 'Sb': Sb[:k], 'Vb': Vb[:k, :]}
    return compressed_data


def decompress_image_svd(compressed_data):
    """Reconstructs an image from its compressed SVD representation."""
    channels = []
    for color in ['r', 'g', 'b']:
        U = compressed_data['U' + color]
        S = np.diag(compressed_data['S' + color])
        V = compressed_data['V' + color]
        channel = np.dot(U, np.dot(S, V))
        channel_clipped = np.clip(channel, 0, 255).astype(np.uint8)
        channels.append(channel_clipped)
    return np.stack(channels, axis=-1)

def save_compressed_data(filepath, data):
    """Saves compressed image data using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_compressed_data(filepath):
    """Loads compressed image data using pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Image compression using SVD")
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True)
    parser.add_argument("--method", choices=["numpy", "simple", "advanced"], required=False)
    parser.add_argument("--compression", type=int, default=2)
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    if args.mode == "compress":
        image = np.array(Image.open(args.in_file))
        compressed_data = compress_image_svd(image, args.method, args.compression)
        save_compressed_data(args.out_file, compressed_data)
    elif args.mode == "decompress":
        compressed_data = load_compressed_data(args.in_file)
        decompressed_image = decompress_image_svd(compressed_data)
        Image.fromarray(decompressed_image).save(args.out_file)

if __name__ == "__main__":
    main()