# Introduction au Calcul Haute Performance sur GPU

## 1\. Introduction : Du Calcul Séquentiel au Parallélisme 

L'informatique traditionnelle repose sur le processeur central (CPU), conçu pour traiter des séries d'instructions hétérogènes avec une rapidité d'exécution (latence) minimale. Cependant, l'avènement du *Deep Learning* et du traitement d'image a imposé des charges de travail différentes : le traitement simultané de gigaoctets de données.

Ce cours introduit le **calcul GPU** (Graphics Processing Unit), une approche qui délaisse la vitesse pure d'exécution unitaire au profit d'un parallélisme massif.

### 1.1 Comparaison Architecturale : Latence versus Débit

La différence fondamentale entre CPU et GPU réside dans la manière dont les puces investissent leurs ressources (transistors) :

  * **CPU (Optimisé pour la Latence) :** Conçu pour que *chaque instruction* se termine le plus vite possible. Il dispose de peu de cœurs, mais chacun est puissant, doté de larges caches et d'unités de prédiction complexes pour gérer la logique séquentielle.
  * **GPU (Optimisé pour le Débit) :** Sacrifie la complexité individuelle pour maximiser le nombre d'unités de calcul (ALU). Il ne cherche pas à aller vite pour *une* tâche, mais à en exécuter des milliers simultanément.

> **💡 L'Analogie : Le Professeur et la Classe**
>
>   * **Le CPU est un Professeur de Mathématiques émérite (ex: Einstein).** Il est brillant et rapide. Il peut résoudre des intégrales complexes en un clin d'œil. Mais s'il doit corriger 10 000 copies d'addition simple, il devra les faire l'une après l'autre. Cela prendra des heures.
>   * **Le GPU est une classe de 1 000 élèves de primaire.** Individuellement, ils sont lents et ne savent faire que des opérations simples. Mais si vous distribuez les 10 000 copies, ils peuvent en corriger 1 000 à la fois. Le travail est fini en quelques secondes.

### 1.2 Traduction en CUDA

En CUDA, on distingue le code qui tourne sur le CPU (**Host**) de celui qui tourne sur le GPU (**Device**).

```cpp
// Code CPU (Le Professeur)
void main() {
    // Prépare les données et lance l'ordre à la classe
    monKernel<<<...>>>(...); 
}

// Code GPU (Les Élèves)
// Le mot-clé "__global__" indique que cette fonction est exécutée sur le GPU
__global__ void monKernel(float* data) {
    // Instruction simple exécutée par des milliers de threads
}
```



## 2\. Architecture Matérielle et Modèle d'Exécution

Pour exploiter toute la puissance du GPU, CUDA décompose l'exécution en une hiérarchie stricte. Une mauvaise gestion de cette structure peut fortement impacter les performances.

### 2.1 Hiérarchie Logique : Grille, Blocs et Threads

CUDA organise les "ouvriers" en trois niveaux :

1.  **La Grille (Grid) :** L'ensemble du problème à résoudre (ex: une image entière).
2.  **Le Bloc (Thread Block) :** Un sous-groupe de la grille. Les threads d'un même bloc peuvent communiquer via une mémoire partagée rapide.
3.  **Le Thread (Fil d'exécution) :** L'unité fondamentale qui traite un seul point de donnée (ex: un pixel).

> **💡 L'Analogie : L'Organisation de l'École**
>
>   * **La Grille** est l'école entière mobilisée pour un examen.
>   * **Le Bloc** est une salle de classe spécifique. Les élèves d'une même salle peuvent se parler (mémoire partagée), mais ne peuvent pas copier sur les élèves de la salle voisine.
>   * **Le Thread** est un élève unique assis à son bureau.

### 2.2 Traduction en CUDA : Le Calcul de Coordonnées

Chaque thread doit savoir "qui il est" pour savoir "quelle donnée traiter". Il calcule ses coordonnées uniques ($x, y$) à partir de son index dans le bloc et de la position du bloc dans la grille.

```cpp
__global__ void imageKernel(int* image, int width) {
    // Qui suis-je ? (Calcul de l'index global)
    // blockIdx.x : Numéro de ma salle de classe
    // blockDim.x : Nombre d'élèves par salle
    // threadIdx.x : Mon numéro de place dans la salle
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Je travaille uniquement si je suis dans les limites de l'image
    if (x < width && y < height) {
        // Traitement de MON pixel unique
    }
}
```

### 2.3 L'Unité Réelle : Le Warp et la Divergence

Physiquement, le GPU n'exécute pas les threads un par un, mais par groupes de 32 appelés **Warps**. Ils suivent le modèle SIMT (*Single Instruction, Multiple Threads*) : ils doivent tous exécuter la même instruction au même moment.

**Le Piège : La Divergence**
Si vous mettez un `if-else` dans votre code, et que la moitié du Warp va dans le `if` et l'autre dans le `else`, le GPU doit exécuter les deux branches séquentiellement, divisant la performance par deux.

> **💡 L'Analogie : La Dictée**
> Le professeur (l'unité de contrôle) dicte à une rangée d'élèves (un Warp).
>
>   * *Cas idéal :* "Écrivez tous le mot 'Chat'". Tous écrivent en même temps.
>   * *Divergence :* "Si vous avez un stylo bleu, écrivez 'Chat', sinon écrivez 'Chien'". Le prof doit d'abord faire écrire ceux au stylo bleu (les autres attendent), puis ceux au stylo rouge. On perd du temps.

-----

## 3\. Gestion de la Mémoire : Le Nerf de la Guerre

Dans une application GPU, la puissance de calcul brute est rarement le facteur limitant. Le véritable goulot d'étranglement est la gestion de la mémoire. Une application mal optimisée peut passer l'essentiel de son temps à attendre l'arrivée des données.

### 3.1 Le Goulot d'Étranglement : le Bus PCIe

L'architecture repose sur la séparation physique de deux espaces mémoires (Host RAM et Device VRAM), connectés par le bus PCI Express (PCIe). La bande passante du bus PCIe est considérablement plus lente (ex: 16-32 Go/s pour une liaison x16) que la bande passante interne de la mémoire du GPU (ex: 900 Go/s pour une architecture Volta). Cette disparité crée le goulot d'étranglement PCIe qui doit être contourné.

La stratégie d'optimisation fondamentale consiste à minimiser les opérations de Transfert Host $\leftrightarrow$ Device : il est impératif d'envoyer la totalité des données d'entrée au début, d'exécuter le calcul intensif sur le Device, puis de ne rapatrier que le résultat final 4

> **💡 L'Analogie : La Bibliothèque et la Salle de Classe**
>* **La RAM CPU (Host)** est la Bibliothèque Universitaire (Source).
>* **La VRAM GPU (Device)** est la Salle d'Examen (Travail).
>* **Le Bus PCIe** est la Camionnette de Livraison.
>
>Le coût majeur de l'opération est lié à la latence de chaque transfert (le temps d'attente pour charger et décharger la camionnette). Pour maximiser l'efficacité (débit), il ne faut jamais envoyer une camionnette pour un seul livre (petit transfert). Il faut consolider les besoins en remplissant la camionnette au maximum de sa capacité avec toutes les données requises, et n'effectuer qu'un seul aller-retour entre l'Hôte et le Device.

### 3.2 Traduction en CUDA : Allocation et Transfert

La gestion mémoire ressemble au C standard (`malloc`, `memcpy`) mais avec le préfixe `cuda`.

```cpp
void gestionMemoire(int imageSize) {
    unsigned char *h_image, *d_image;

    // 1. Allocation CPU (Host)
    h_image = (unsigned char*)malloc(imageSize);

    // 2. Allocation GPU (Device) - On prépare le tableau noir
    cudaMalloc((void**)&d_image, imageSize);

    // 3. Transfert CPU -> GPU (La camionnette part)
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    // ... Exécution du Kernel sur d_image ...

    // 4. Récupération GPU -> CPU (La camionnette revient)
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
}
```


## 4\. Application : Filtre Sépia

Pour prouver la pertinence du GPU, nous appliquons un filtre Sépia sur une image 4K (8 millions de pixels). C'est un problème qui se prête idéalement au parallélisme massif.

### 4.1 Le Kernel Complet

Voici le cœur du programme. Notez l'absence de boucle `for` : la boucle est remplacée par la grille de threads.

```cpp
__global__ void sepiaKernel(unsigned char* image, int width, int height) {
    // Calcul des coordonnées (L'élève trouve sa place)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Index linéarisé pour accéder à la mémoire 1D
        int tid = (y * width + x) * 3; // 3 canaux (RGB)

        // Lecture des couleurs (L'élève lit sa donnée)
        float r = image[tid];
        float g = image[tid+1];
        float b = image[tid+2];

        // Calcul Sépia (L'élève fait son calcul)
        float new_r = (r * 0.393) + (g * 0.769) + (b * 0.189);
        float new_g = (r * 0.349) + (g * 0.686) + (b * 0.168);
        float new_b = (r * 0.272) + (g * 0.534) + (b * 0.131);

        // Écriture (L'élève note le résultat)
        image[tid] = (unsigned char)min(255.0f, new_r);
        image[tid+1] = (unsigned char)min(255.0f, new_g);
        image[tid+2] = (unsigned char)min(255.0f, new_b);
    }
}
```

### 4.2 Lancement depuis le CPU

Comment organiser notre armée de threads ? Pour une image, on utilise généralement des blocs carrés de 16x16 threads (256 threads par bloc).

```cpp
int main() {
    // ... Allocation et Transferts (voir section 3.2) ...

    // Définition de la taille de l'équipe (Bloc)
    dim3 threadsPerBlock(16, 16); 

    // Calcul du nombre d'équipes nécessaires (Grille)
    // On divise la taille de l'image par 16, en arrondissant au supérieur
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    // Lancement de l'assaut
    sepiaKernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
    
    // Attente de la fin
    cudaDeviceSynchronize();
}
```

### 4.3\. Analyse de Performance

Est-ce que tout cet effort de programmation en vaut la peine ? Voici une comparaison typique pour le traitement d'une image haute définition.

| Métrique | CPU (Intel Core i7) | GPU (NVIDIA Tesla T4) | Gain |
| :--- | :--- | :--- | :--- |
| **Méthode** | Boucle séquentielle | 8 millions de threads parallèles | - |
| **Temps de calcul** | \~250 ms | \~3 ms | **x80** |
| **Philosophie** | Une Ferrari faisant 8 millions d'allers-retours | Un train de marchandises transportant tout d'un coup | - |

### 4.4\. Conclusion
Le GPU n'est pas "plus rapide" au sens où il court plus vite (fréquence en MHz souvent inférieure au CPU). Il est plus performant car il est **plus large**. Pour des tâches massives comme le traitement d'image ou l'IA, le paradigme CUDA permet de transformer un problème temporel (attendre la fin de la boucle) en un problème spatial (occuper toute la surface de la puce).
