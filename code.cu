%%writefile sepia_simple.cu

/*
 * TP HPC 2025 : Introduction à CUDA
 * Sujet : Comparaison CPU vs GPU sur un Filtre Sépia
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h> // Pour mesurer le temps simplement

// On travaille sur une grande image (4K) pour voir la différence
#define WIDTH 3840
#define HEIGHT 2160
#define CHANNELS 3 // Rouge, Vert, Bleu

// Nombre de fois qu'on répète le calcul pour avoir une mesure fiable
#define ITERATIONS 100 

// --------------------------------------------------------
// 1. LE CODE GPU (KERNEL)
// Cette fonction est exécutée par des MILLIERS de threads en même temps.
// Chaque thread s'occupe d'UN SEUL pixel.
// --------------------------------------------------------
__global__ void sepiaKernel(unsigned char* image, int width, int height) {
    
    // Le thread calcule ses coordonnées (x, y) dans l'image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // On vérifie qu'on est bien dans l'image
    if (x < width && y < height) {
        
        // On calcule la position dans le tableau 1D
        int index = (y * width + x) * CHANNELS;

        // Lecture des couleurs
        float r = image[index];
        float g = image[index+1];
        float b = image[index+2];

        // Calcul Sépia (Formule mathématique)
        float new_r = (r * 0.393f) + (g * 0.769f) + (b * 0.189f);
        float new_g = (r * 0.349f) + (g * 0.686f) + (b * 0.168f);
        float new_b = (r * 0.272f) + (g * 0.534f) + (b * 0.131f);

        // Ecriture (fminf garde la valeur sous 255)
        image[index]   = (unsigned char)fminf(255.0f, new_r);
        image[index+1] = (unsigned char)fminf(255.0f, new_g);
        image[index+2] = (unsigned char)fminf(255.0f, new_b);
    }
}

// --------------------------------------------------------
// 2. LE CODE CPU (CLASSIQUE)
// Cette fonction utilise une boucle "for". Elle traite les pixels un par un.
// --------------------------------------------------------
void sepiaCPU(unsigned char* image, int width, int height) {
    int totalBytes = width * height * CHANNELS;
    
    // Boucle sur tous les pixels (Séquentiel)
    for (int i = 0; i < totalBytes; i += 3) {
        float r = image[i];
        float g = image[i+1];
        float b = image[i+2];

        image[i]   = (unsigned char)fminf(255.0f, (r * 0.393f) + (g * 0.769f) + (b * 0.189f));
        image[i+1] = (unsigned char)fminf(255.0f, (r * 0.349f) + (g * 0.686f) + (b * 0.168f));
        image[i+2] = (unsigned char)fminf(255.0f, (r * 0.272f) + (g * 0.534f) + (b * 0.131f));
    }
}

// --------------------------------------------------------
// 3. LE PROGRAMME PRINCIPAL (MAIN)
// --------------------------------------------------------
int main() {
    size_t size = WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char);
    printf("Traitement d'une image de %dx%d pixels (x%d fois)...\n", WIDTH, HEIGHT, ITERATIONS);

    // Allocation mémoire (RAM de l'ordinateur)
    unsigned char *img_cpu = (unsigned char*)malloc(size);
    unsigned char *img_gpu = (unsigned char*)malloc(size); // Copie pour le GPU
    
    // On remplit l'image avec du bruit aléatoire pour tester
    for(size_t i=0; i<size; i++) {
        unsigned char valeur = rand() % 256;
        img_cpu[i] = valeur;
        img_gpu[i] = valeur;
    }

    // --- TEST 1 : CPU ---
    printf("Lancement CPU... ");
    clock_t start = clock(); // Top départ
    
    // On répète le calcul 100 fois pour bien mesurer l'effort
    for(int k=0; k<ITERATIONS; k++) {
        sepiaCPU(img_cpu, WIDTH, HEIGHT);
    }
    
    clock_t end = clock();   // Top fin
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Fini en %.3f secondes.\n", cpu_time);


    // --- TEST 2 : GPU ---
    printf("Lancement GPU... ");
    
    unsigned char *d_img;
    // 1. Allocation mémoire sur la carte graphique (VRAM)
    CHECK(cudaMalloc((void**)&d_img, size));
    
    // 2. Copie des données : PC vers Carte Graphique
    CHECK(cudaMemcpy(d_img, img_gpu, size, cudaMemcpyHostToDevice));

    // 3. Configuration : On découpe l'image en blocs de 16x16 threads
    dim3 threadsParBloc(16, 16);
    dim3 nombreDeBlocs((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // 4. Chrono GPU (Plus précis pour le GPU)
    cudaEvent_t depart, fin;
    cudaEventCreate(&depart); cudaEventCreate(&fin);
    
    cudaEventRecord(depart);
    
    // *** LANCEMENT DU KERNEL (x100 fois) ***
    for(int k=0; k<ITERATIONS; k++) {
        sepiaKernel<<<nombreDeBlocs, threadsParBloc>>>(d_img, WIDTH, HEIGHT);
    }
    
    cudaEventRecord(fin);
    
    // On attend que le GPU ait fini
    CHECK(cudaDeviceSynchronize()); 

    // Calcul du temps GPU
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, depart, fin);
    double gpu_time = milliseconds / 1000.0;
    printf("Fini en %.3f secondes.\n", gpu_time);

    // 5. Récupération du résultat : Carte Graphique vers PC
    CHECK(cudaMemcpy(img_gpu, d_img, size, cudaMemcpyDeviceToHost));

    // --- CONCLUSION ---
    printf("\n=== RESULTAT ===\n");
    printf("Le GPU est %.1f fois plus rapide que le CPU.\n", cpu_time / gpu_time);

    // Nettoyage
    cudaFree(d_img);
    free(img_cpu); free(img_gpu);
    return 0;
}