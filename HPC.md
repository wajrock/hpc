# Introduction au Calcul Haute Performance sur GPU

## 1\. Introduction : Du Calcul Séquentiel au Parallélisme 

Pendant des décennies, l'informatique a reposé sur un seul maître à bord : le **CPU** (Processeur Central). Imaginez-le comme un gestionnaire ultra-rapide, capable de passer d'un email à un fichier Excel puis à une page web en un éclair. Sa force, c'est la **logique séquentielle** : il traite les problèmes complexes les uns après les autres.

Cependant, l'arrivée des jeux vidéo 3D, du traitement d'image haute définition et plus récemment de l'Intelligence Artificielle (*Deep Learning*) a changé la donne. Ici, il ne s'agit plus de résoudre une équation complexe, mais de traiter **des millions de petites données identiques simultanément** (par exemple : changer la couleur de 8 millions de pixels sur un écran 4K).

Face à cette charge, le CPU s'essouffle. Ce cours vous introduit au **calcul GPU** (Graphics Processing Unit). C'est une approche qui abandonne la "vitesse pure sur une tâche" au profit du **parallélisme massif** : faire moins vite individuellement, mais faire tout en même temps.


### 1.1 Comparaison Architecturale : Latence versus Débit

La différence fondamentale entre CPU et GPU réside dans la manière dont les puces investissent leurs ressources (transistors) :

  * **CPU (Optimisé pour la Latence) :** Conçu pour que *chaque instruction* se termine le plus vite possible. Il dispose de peu de cœurs, mais chacun est puissant, doté de larges caches et d'unités de prédiction complexes pour gérer la logique séquentielle.

  * **GPU (Optimisé pour le Débit) :** Sacrifie la complexité individuelle pour maximiser le nombre d'unités de calcul (ALU). Il ne cherche pas à aller vite pour *une* tâche, mais à en exécuter des milliers simultanément.

> **💡 L'Analogie : Le Professeur et la Classe**
>
>   * **Le CPU est un Professeur de Mathématiques émérite (ex: Einstein).** Il est brillant et rapide. Il peut résoudre des intégrales complexes en un clin d'œil. Mais s'il doit corriger 10 000 copies d'addition simple, il devra les faire l'une après l'autre. Cela prendra des heures.
>   * **Le GPU est une classe de 1 000 élèves de primaire.** Individuellement, ils sont lents et ne savent faire que des opérations simples. Mais si vous distribuez les 10 000 copies, ils peuvent en corriger 1 000 à la fois. Le travail est fini en quelques secondes.

### 1.2 Traduction en CUDA

En programmation CUDA, nous écrivons tout dans le même fichier (extension `.cu`), mais il faut comprendre que deux mondes physiquement séparés cohabitent :

1.  **Le Host (Hôte) :** C'est votre **CPU**. Il joue le rôle de **Chef d'Orchestre**. Il ne fait pas le calcul intensif lui-même, mais il gère la logistique : il prépare les données, les envoie au GPU et donne le signal de départ.
2.  **Le Device (Périphérique) :** C'est votre **GPU**. C'est l'**Usine**. Il attend les ordres et les données pour lancer ses milliers d'ouvriers.

> **💡 L'Analogie : Les Consignes vs L'Exercice**
>
> * **Le Code Host (CPU)** correspond aux **consignes orales** du professeur : *"Prenez une feuille, recopiez l'exercice au tableau, vous avez 1 heure."*
> * **Le Code Device (GPU)** correspond à **l'énoncé de l'exercice** lui-même : *"Calculez la racine carrée de x."* Chaque élève (Thread) va appliquer cet énoncé à sa propre feuille.

Pour distinguer ces deux mondes dans le code, CUDA utilise un mot-clé spécial : `__global__`.

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


Voici la restructuration complète de la section 4, intégrant les résultats réels du benchmark (qui sont excellents \!), les analogies demandées et une présentation académique soignée.

-----

## 4\. Application Pratique : Implémentation Parallèle d'un Filtre Sépia

Cette section constitue l'étude de cas où tous les concepts architecturaux et de gestion mémoire sont appliqués. Nous utilisons l'application d'un filtre Sépia, un algorithme parfaitement adapté au GPU car intrinsèquement massivement parallèle.

### 4.1 Justification du Choix d'Algorithme

Le traitement d'image est un cas idéal car chaque pixel est indépendant (*embarrassingly parallel*). L'algorithme Sépia consiste à appliquer une transformation matricielle à chaque pixel RGB pour obtenir un effet "vieille photo".

**Stratégie de Projection :** Nous appliquons une stratégie de mappage **un à un** (1:1) : **Un Thread CUDA est responsable du traitement d'Un seul Pixel de l'image.**

### 4.2 Défi Technique : Le Mapping 2D vers 1D

Bien qu'une image soit une grille 2D (lignes et colonnes), la mémoire vidéo (VRAM) la stocke comme un tableau linéaire continu (1D). Chaque thread doit donc calculer son adresse unique dans ce "ruban" mémoire.

> **💡 L'Analogie : La Bibliothèque**
> Imaginez une bibliothèque avec 10 étagères ($y$) de 100 livres ($x$) chacune.
> Si vous voulez le 5ème livre de la 3ème étagère, combien de livres y a-t-il avant lui ?
>
>   * Vous devez passer les 2 étagères complètes précédentes ($y \times \text{largeur}$).
>   * Plus les 5 livres de l'étagère actuelle ($+ x$).
>
> **Formule :** $\text{Index} = (y \times \text{Largeur}) + x$

### 4.3 Le Noyau CUDA et l'Optimisation d'Accès

Le code du noyau (`sepiaKernel`) ne contient aucune boucle `for`. Il décrit l'action d'un seul thread sur un seul pixel.

#### 4.3.1 Le Code du Kernel (Device)

```cpp
__global__ void sepiaKernel(unsigned char* image, int width, int height) {
    // 1. Calcul des coordonnées globales 2D (L'élève trouve sa place)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Vérification des limites (Guard)
    if (x < width && y < height) {
        
        // 3. Conversion 2D -> 1D (Mapping mémoire)
        // Multiplié par 3 car chaque pixel a 3 composantes (R, G, B)
        int tid = (y * width + x) * 3;

        // Lecture (Accès coalescé optimisé)
        float r = image[tid];
        float g = image[tid+1];
        float b = image[tid+2];

        // Calcul Sépia (Opération arithmétique SIMT)
        float new_r = (r * 0.393f) + (g * 0.769f) + (b * 0.189f);
        float new_g = (r * 0.349f) + (g * 0.686f) + (b * 0.168f);
        float new_b = (r * 0.272f) + (g * 0.534f) + (b * 0.131f);

        // Écriture (Saturation à 255 pour éviter les débordements visuels)
        image[tid]   = (unsigned char)fminf(255.0f, new_r);
        image[tid+1] = (unsigned char)fminf(255.0f, new_g);
        image[tid+2] = (unsigned char)fminf(255.0f, new_b);
    }
}
```

#### 4.3.2 Configuration et Lancement (Host)

Le CPU doit définir la taille de la grille (combien de blocs ?) pour couvrir toute l'image.

Pour mieux comprendre imaginez que vous devez transporter 100 élèves (pixels) et vos bus (blocs) ont 16 places. $100 / 16 = 6.25$. Si vous prenez 6 bus, 4 élèves ne pourrons pas monter. Il faut donc commander **7 bus** (arrondi supérieur), même si le dernier part partiellement vide.

```cpp
// Configuration standard : Blocs carrés de 16x16 threads
dim3 threadsPerBlock(16, 16); 

// Calcul du nombre de blocs (Arrondi supérieur)
dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

// Lancement du Kernel
sepiaKernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
cudaDeviceSynchronize(); // Attente de la fin du calcul
```

### 4.4 Analyse de Performance Réelle

Pour valider l'approche, nous avons effectué un test de charge ("Stress Test") sur une image 4K ($3840 \times 2160$) traitée 100 fois consécutivement.

**Résultats Expérimentaux (Google Collab T4 GPU) :**

| Métrique | CPU (Séquentiel) | GPU (Parallèle) |
| :--- | :--- | :--- |
| **Temps Total** | 8 125 ms | 49 ms |
| **Temps par Image** | \~81 ms | \~0.5 ms |
| **Débit** | \~0.1 Gigapixels/s | \~16.9 Gigapixels/s |

**Facteur d'Accélération (Speedup) : $\times 165.4$**

**Interprétation :**
Le GPU traite l'image **165 fois plus vite** que le CPU. Là où le CPU traite les pixels un par un séquentiellement (latence cumulative), le GPU lance 8.3 millions de threads simultanément.

### 4.5. Conclusion 

En résumé, il ne faut pas retenir que le GPU est "plus rapide" que le CPU (sa fréquence en MHz est souvent inférieure), mais qu'il est **massivement parallèle**.

Ce cours vous a invité à un changement de philosophie fondamental : nous sommes passés d'une architecture optimisée pour la **latence** (exécuter une tâche le plus vite possible) à une architecture dédiée au **débit** (exécuter des milliers de tâches simultanément).

Maîtriser CUDA, ce n'est pas seulement apprendre une nouvelle syntaxe. C'est comprendre comment transformer un problème temporel (attendre la fin d'une boucle) en un problème spatial (occuper toute la surface de la puce avec des milliers de threads). C'est cette capacité à "diviser pour régner" à grande échelle qui rend aujourd'hui possibles les avancées majeures en *Deep Learning* et en simulation scientifique.

