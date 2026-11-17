#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <turbojpeg.h>
#include <vector>

using namespace std;

constexpr int BLOCK_SIZE = 32;

struct SVDBlock {
    vector<float> sigma;
    vector<int8_t> U, V;
};

#pragma pack(push, 1)
struct SVDHeader {
    char magic[4];
    uint32_t width, height, blockSize, K, totalBlocks;
};
#pragma pack(pop)

bool decodeJPEG(const string& path, vector<uint8_t>& pixels, int& w, int& h, streamsize& origSize) {
    ifstream in(path, ios::binary | ios::ate);
    if (!in) { cerr << "Error: Cannot open " << path << "\n"; return false; }
    origSize = in.tellg();
    in.seekg(0, ios::beg);
    vector<unsigned char> buf(origSize);
    if (!in.read(reinterpret_cast<char*>(buf.data()), origSize)) return false;

    tjhandle tj = tjInitDecompress();
    if (!tj) { cerr << "Error: " << tjGetErrorStr() << "\n"; return false; }
    int subsamp = 0, cs = 0;
    if (tjDecompressHeader3(tj, buf.data(), origSize, &w, &h, &subsamp, &cs) < 0) {
        cerr << "Error: " << tjGetErrorStr() << "\n"; tjDestroy(tj); return false;
    }
    pixels.resize(static_cast<size_t>(w) * h);
    int ret = tjDecompress2(tj, buf.data(), origSize, pixels.data(), w, 0, h, TJPF_GRAY, 0);
    tjDestroy(tj);
    if (ret < 0) { cerr << "Error: " << tjGetErrorStr() << "\n"; return false; }
    return true;
}

bool encodeJPEG(const string& path, const uint8_t* pixels, int w, int h, int qual, unsigned long& outSize) {
    tjhandle tj = tjInitCompress();
    if (!tj) { cerr << "Error: " << tjGetErrorStr() << "\n"; return false; }
    unsigned char* jpegBuf = nullptr;
    if (tjCompress2(tj, pixels, w, 0, h, TJPF_GRAY, &jpegBuf, &outSize, TJSAMP_GRAY, qual, 0) < 0) {
        cerr << "Error: " << tjGetErrorStr() << "\n"; tjDestroy(tj); return false;
    }
    ofstream out(path, ios::binary);
    if (!out) {
        cerr << "Error: Cannot write " << path << "\n";
        tjFree(jpegBuf); tjDestroy(tj); return false;
    }
    out.write(reinterpret_cast<const char*>(jpegBuf), outSize);
    tjFree(jpegBuf); tjDestroy(tj);
    return true;
}

void computeBlockSVD(const float A[BLOCK_SIZE][BLOCK_SIZE], int K, SVDBlock& out) {
    out.sigma.resize(K);
    out.U.assign(K * BLOCK_SIZE, 0);
    out.V.assign(K * BLOCK_SIZE, 0);

    // 1. Symmetric matrix S = A^T * A
    double S[BLOCK_SIZE][BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        for (int j = i; j < BLOCK_SIZE; ++j) {
            double sum = 0.0;
            for (int k = 0; k < BLOCK_SIZE; ++k) sum += static_cast<double>(A[k][i]) * A[k][j];
            S[i][j] = S[j][i] = sum;
        }
    }

    // 2. Initialize V = I
    double V[BLOCK_SIZE][BLOCK_SIZE] = {0};
    for (int i = 0; i < BLOCK_SIZE; ++i) V[i][i] = 1.0;

    // 3. Jacobi sweeps
    for (int sweep = 0; sweep < 25; ++sweep) {
        double off_diag = 0.0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
            for (int j = i + 1; j < BLOCK_SIZE; ++j) off_diag += abs(S[i][j]);
        if (off_diag < 1e-7) break;

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = i + 1; j < BLOCK_SIZE; ++j) {
                if (abs(S[i][j]) < 1e-13) continue;
                double tau = (S[j][j] - S[i][i]) / (2.0 * S[i][j]);
                double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                        : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                double c = 1.0 / sqrt(1.0 + t * t), s = t * c;
                double h = t * S[i][j];
                S[i][i] -= h; S[j][j] += h; S[i][j] = S[j][i] = 0.0;

                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    if (k != i && k != j) {
                        double ski = S[k][i], skj = S[k][j];
                        S[k][i] = S[i][k] = c * ski - s * skj;
                        S[k][j] = S[j][k] = s * ski + c * skj;
                    }
                    double vki = V[k][i], vkj = V[k][j];
                    V[k][i] = c * vki - s * vkj;
                    V[k][j] = s * vki + c * vkj;
                }
            }
        }
    }

    // 4. Eigenvalues to singular values & sorting
    struct Comp { double sig; int col; };
    Comp comps[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i)
        comps[i] = { (S[i][i] > 0.0) ? sqrt(S[i][i]) : 0.0, i };
    sort(comps, comps + BLOCK_SIZE, [](const Comp& a, const Comp& b) { return a.sig > b.sig; });

    // 5. Extract top K components with quantization
    for (int r = 0; r < K; ++r) {
        double sig = comps[r].sig;
        out.sigma[r] = static_cast<float>(sig);
        int vc = comps[r].col;

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            double val = max(-1.0, min(1.0, V[k][vc]));
            out.V[r * BLOCK_SIZE + k] = static_cast<int8_t>(round(val * 127.0));
        }

        if (sig > 1e-7) {
            double u[BLOCK_SIZE], unorm = 0.0;
            for (int row = 0; row < BLOCK_SIZE; ++row) {
                double sum = 0.0;
                for (int col = 0; col < BLOCK_SIZE; ++col) sum += static_cast<double>(A[row][col]) * V[col][vc];
                u[row] = sum / sig;
                unorm += u[row] * u[row];
            }
            unorm = sqrt(unorm);
            if (unorm > 1e-7) {
                for (int row = 0; row < BLOCK_SIZE; ++row) {
                    double val = max(-1.0, min(1.0, u[row] / unorm));
                    out.U[r * BLOCK_SIZE + row] = static_cast<int8_t>(round(val * 127.0));
                }
            }
        }
    }
}

void extractBlock(const uint8_t* img, int w, int h, int bx, int by, float blk[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int r = 0; r < BLOCK_SIZE; ++r) {
        int y = min(by * BLOCK_SIZE + r, h - 1);
        for (int c = 0; c < BLOCK_SIZE; ++c)
            blk[r][c] = static_cast<float>(img[y * w + min(bx * BLOCK_SIZE + c, w - 1)]);
    }
}

void sequentialSVD(const uint8_t* img, int w, int h, int K, vector<SVDBlock>& blocks) {
    int nx = (w + BLOCK_SIZE - 1) / BLOCK_SIZE, ny = (h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks.resize(nx * ny);
    for (int b = 0; b < nx * ny; ++b) {
        float blk[BLOCK_SIZE][BLOCK_SIZE];
        extractBlock(img, w, h, b % nx, b / nx, blk);
        computeBlockSVD(blk, K, blocks[b]);
    }
}
//core logic
void parallelSVD(const uint8_t* img, int w, int h, int K, vector<SVDBlock>& blocks) {
    int nx = (w + BLOCK_SIZE - 1) / BLOCK_SIZE, ny = (h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks.resize(nx * ny);
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nx * ny; ++b) {
        float blk[BLOCK_SIZE][BLOCK_SIZE];
        extractBlock(img, w, h, b % nx, b / nx, blk);
        computeBlockSVD(blk, K, blocks[b]);
    }
}

bool writeSVDFile(const string& path, int w, int h, int K, int totalBlocks, const vector<SVDBlock>& blocks) {
    ofstream out(path, ios::binary);
    if (!out) return false;
    SVDHeader hdr = { {'S','V','D','1'}, (uint32_t)w, (uint32_t)h, BLOCK_SIZE, (uint32_t)K, (uint32_t)totalBlocks };
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (const auto& b : blocks) {
        out.write(reinterpret_cast<const char*>(b.sigma.data()), K * sizeof(float));
        out.write(reinterpret_cast<const char*>(b.U.data()), K * BLOCK_SIZE * sizeof(int8_t));
        out.write(reinterpret_cast<const char*>(b.V.data()), K * BLOCK_SIZE * sizeof(int8_t));
    }
    return out.good();
}

bool readSVDAndReconstruct(const string& path, int& outW, int& outH, int& outK, vector<uint8_t>& pixels) {
    ifstream in(path, ios::binary);
    if (!in) return false;
    SVDHeader hdr;
    if (!in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) return false;
    if (hdr.magic[0] != 'S' || hdr.magic[1] != 'V' || hdr.magic[2] != 'D' || hdr.magic[3] != '1' ||
        hdr.blockSize != BLOCK_SIZE || hdr.K == 0 || hdr.K > BLOCK_SIZE) return false;

    outW = hdr.width; outH = hdr.height; outK = hdr.K;
    int nx = (outW + BLOCK_SIZE - 1) / BLOCK_SIZE, ny = (outH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (hdr.totalBlocks != static_cast<uint32_t>(nx * ny)) return false;

    pixels.assign(static_cast<size_t>(outW) * outH, 0);
    vector<float> sigma(outK);
    vector<int8_t> U(outK * BLOCK_SIZE), V(outK * BLOCK_SIZE);

    for (int b = 0; b < nx * ny; ++b) {
        if (!in.read(reinterpret_cast<char*>(sigma.data()), outK * sizeof(float)) ||
            !in.read(reinterpret_cast<char*>(U.data()), outK * BLOCK_SIZE * sizeof(int8_t)) ||
            !in.read(reinterpret_cast<char*>(V.data()), outK * BLOCK_SIZE * sizeof(int8_t)))
            return false;

        float A[BLOCK_SIZE][BLOCK_SIZE] = {0};
        for (int r = 0; r < outK; ++r) {
            float s = sigma[r];
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float ui = static_cast<float>(U[r * BLOCK_SIZE + i]) / 127.0f;
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    A[i][j] += s * ui * (static_cast<float>(V[r * BLOCK_SIZE + j]) / 127.0f);
            }
        }

        int bx = b % nx, by = b / nx;
        for (int r = 0; r < BLOCK_SIZE && by * BLOCK_SIZE + r < outH; ++r) {
            int y = by * BLOCK_SIZE + r;
            for (int c = 0; c < BLOCK_SIZE && bx * BLOCK_SIZE + c < outW; ++c) {
                int ival = static_cast<int>(round(A[r][c]));
                pixels[y * outW + (bx * BLOCK_SIZE + c)] = static_cast<uint8_t>(clamp(ival, 0, 255));
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input.jpg> <compressed.svd> <reconstructed.jpg> <K>\n";
        return 1;
    }
    string inJpg = argv[1], svdPath = argv[2], recJpg = argv[3];
    int K = stoi(argv[4]);
    if (K < 1 || K > BLOCK_SIZE) {
        cerr << "Error: K must be between 1 and " << BLOCK_SIZE << "\n";
        return 1;
    }

    vector<uint8_t> rawPixels;
    int w = 0, h = 0;
    streamsize origJpgSize = 0;
    if (!decodeJPEG(inJpg, rawPixels, w, h, origJpgSize)) return 1;

    size_t rawSize = static_cast<size_t>(w) * h;
    int totalBlocks = ((w + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    vector<SVDBlock> seqBlocks, parBlocks;
    double t0 = omp_get_wtime();
    sequentialSVD(rawPixels.data(), w, h, K, seqBlocks);
    double seqTime = omp_get_wtime() - t0;

    double t1 = omp_get_wtime();
    parallelSVD(rawPixels.data(), w, h, K, parBlocks);
    double parTime = omp_get_wtime() - t1;

    int threads = omp_get_max_threads();
    double speedup = (parTime > 0.0) ? (seqTime / parTime) : 0.0;
    double efficiency = (threads > 0) ? (speedup / threads * 100.0) : 0.0;

    if (!writeSVDFile(svdPath, w, h, K, totalBlocks, parBlocks)) {
        cerr << "Error: Failed to write SVD file: " << svdPath << "\n";
        return 1;
    }

    ifstream svdCheck(svdPath, ios::binary | ios::ate);
    size_t svdSize = svdCheck ? static_cast<size_t>(svdCheck.tellg()) : 0;
    svdCheck.close();

    vector<uint8_t> recPixels;
    int rw = 0, rh = 0, rk = 0;
    if (!readSVDAndReconstruct(svdPath, rw, rh, rk, recPixels)) {
        cerr << "Error: Failed to decode SVD file: " << svdPath << "\n";
        return 1;
    }

    double sse = 0.0;
    for (size_t i = 0; i < rawSize; ++i) {
        double d = static_cast<double>(rawPixels[i]) - static_cast<double>(recPixels[i]);
        sse += d * d;
    }
    double mse = sse / static_cast<double>(rawSize);
    double psnr = (mse <= 1e-10) ? 99.99 : (10.0 * log10((255.0 * 255.0) / mse));

    unsigned long recJpgSize = 0;
    if (!encodeJPEG(recJpg, recPixels.data(), w, h, 90, recJpgSize)) return 1;

    bool verified = (rw == w && rh == h && rk == K && recPixels.size() == rawPixels.size());
    double cr = (svdSize > 0) ? (static_cast<double>(rawSize) / svdSize) : 0.0;
    double saved = (rawSize > 0) ? ((1.0 - static_cast<double>(svdSize) / rawSize) * 100.0) : 0.0;

    cout << "\n========== SVD IMAGE COMPRESSION ==========\n\n";
    cout << "Image              : " << w << " x " << h << "\n";
    cout << "Block Size         : " << BLOCK_SIZE << " x " << BLOCK_SIZE << "\n";
    cout << "K                  : " << K << "\n";
    cout << "OpenMP Threads     : " << threads << "\n\n";
    cout << fixed << setprecision(3);
    cout << "Sequential SVD Time: " << seqTime << " sec\n";
    cout << "Parallel SVD Time  : " << parTime << " sec\n\n";
    cout << fixed << setprecision(2);
    cout << "Speedup            : " << speedup << "x\n";
    cout << "Efficiency         : " << efficiency << "%\n\n";
    cout << "Raw Image Size     : " << rawSize << " bytes\n";
    cout << "SVD File Size      : " << svdSize << " bytes\n";
    cout << "Compression Ratio  : " << cr << ":1\n";
    cout << "Space Saved        : " << saved << "%\n\n";
    cout << "MSE                : " << mse << "\n";
    cout << "PSNR               : " << psnr << " dB\n\n";
    cout << "Original JPEG Size : " << origJpgSize << " bytes\n";
    cout << "Reconstructed JPEG : " << recJpgSize << " bytes\n\n";
    cout << "Verification       : " << (verified ? "Successful" : "Failed") << "\n\n";
    cout << "============================================\n\n";

    return 0;
}
