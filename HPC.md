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


## 4. Application Concrète : Le Filtre Sépia

Pour démontrer la supériorité du GPU sur des tâches massives, nous allons traiter une image 4K ($3840 \times 2160$ pixels). Cela représente **8,3 millions de pixels**.

Le but est d'appliquer un effet "Sépia" (vieille photo). Pour l'ordinateur, cela signifie lire chaque pixel, mélanger ses canaux Rouge-Vert-Bleu (RGB) selon une formule précise, et réécrire le résultat.


### 4.1. Le Défi : Comprendre la Mémoire 1D

 Nous voyons l'image comme une grille en 2D (lignes et colonnes), mais la mémoire vidéo (VRAM) stocke tout sur une seule ligne continue (1D), comme un immense ruban.

Pour qu'un thread (traitant le pixel $x, y$) trouve sa couleur sur ce ruban, il doit convertir ses coordonnées 2D en index 1D.

> **💡 L'Analogie : La Bibliothèque**
>
> Imaginez une bibliothèque avec 10 étagères ($y$) de 100 livres ($x$) chacune. Si vous voulez le 5ème livre de la 3ème étagère, combien de livres y a-t-il avant lui ?
>
> * Vous devez passer les 2 étagères complètes précédentes ($y \times \text{largeur}$).
> * Plus les 5 livres de l'étagère actuelle ($+ x$).
>
> **La Formule :**
> $$Index = (y \times Largeur) + x$$

### 4.2. Le Code : 8 Millions de Peintres

Voici le **Kernel** (le code exécuté par le GPU). Remarquez qu'il n'y a aucune boucle **for**. Ce code décrit la tâche d'un seul thread pour un seul pixel.

Deux concepts sont souvent difficiles à saisir ici, expliquons-les avant de voir le code :

1. Pourquoi * 3 dans l'index ? Chaque pixel est composé de 3 valeurs : Rouge, Vert, Bleu. Si le thread s'occupe du pixel n°10, il ne doit pas écrire à la case 10 de la mémoire, mais à la case 30 (car les 10 pixels précédents occupent chacun 3 places).

2. Pourquoi min(255, ...) ? Le filtre Sépia a tendance à éclaircir l'image. Si le calcul donne "300", cela dépasse la capacité d'un octet (max 255). Sans cette sécurité, la valeur "déborderait" (300 devient 44) et créerait des points noirs aberrants sur l'image.

```cpp
__global__ void sepiaKernel(unsigned char* image, int width, int height) {
    // --- ÉTAPE 1 : IDENTIFICATION ---
    // Chaque "peintre" (thread) calcule sa position unique sur la toile
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // SÉCURITÉ : On vérifie qu'on ne peint pas hors du cadre
    if (x < width && y < height) {

        // --- ÉTAPE 2 : LOCALISATION MÉMOIRE ---
        // On convertit la position (x,y) en adresse mémoire linéaire.
        // On multiplie par 3 car chaque pixel contient 3 valeurs (R, G, B).
        int tid = (y * width + x) * 3;

        // --- ÉTAPE 3 : LECTURE ---
        // On utilise des 'float' pour ne pas perdre de précision dans les calculs
        float r = image[tid];     // Rouge
        float g = image[tid+1];   // Vert
        float b = image[tid+2];   // Bleu

        // --- ÉTAPE 4 : MÉLANGE (Formule Sépia) ---
        // L'oeil humain est plus sensible au vert, d'où les coefficients différents.
        float new_r = (r * 0.393f) + (g * 0.769f) + (b * 0.189f);
        float new_g = (r * 0.349f) + (g * 0.686f) + (b * 0.168f);
        float new_b = (r * 0.272f) + (g * 0.534f) + (b * 0.131f);

        // --- ÉTAPE 5 : ÉCRITURE ---
        // On borne les valeurs à 255 (min) pour éviter les bugs d'affichage
        image[tid]   = (unsigned char)min(255.0f, new_r);
        image[tid+1] = (unsigned char)min(255.0f, new_g);
        image[tid+2] = (unsigned char)min(255.0f, new_b);
    }
}
```

### 4.3. Le Lancement (Host)

C'est ici que le CPU (le Chef) organise les équipes et lance le travail. Le défi principal est de calculer combien de blocs (équipes) sont nécessaires pour couvrir toute l'image.

Pour comprendre le calcul, **visualisez une sortie scolaire géante** :

> **L'Analogie : La Flotte de Bus**
>
> Imaginez que vous devez transporter tous les élèves de l'école (vos pixels) vers le lieu de l'examen.
>
> * Vous disposez d'une flotte de **bus scolaires** (vos Blocs).
> * Chaque bus a exactement **16 places** (la dimension `threadsPerBlock`).
>
> **L'exemple pratique :** Si vous avez **100 élèves** à transporter :
> 1.  Si vous faites une division simple : $100 / 16 = 6.25$.
> 2.  Si vous commandez **6 bus**, vous transportez 96 élèves et vous en laissez **4 sur le trottoir**.
> 3.  Il est donc impératif de commander **7 bus**, même si le dernier part avec des sièges vides.

En informatique, cette "commande de bus supplémentaire" se traduit par une formule d'arrondi au supérieur :



```cpp
void main() {
    // ... (Allocation mémoire et copie des données faites précédemment) ...

    // 1. Définition de la taille d'un bus (Bloc)
    // 16x16 = 256 threads. C'est un standard efficace sur NVIDIA.
    dim3 threadsPerBlock(16, 16); 

    // 2. Commande du nombre de bus (Grille)
    // On utilise la formule d'arrondi pour couvrir toute l'image
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    // 3. LE DÉPART (Lancement du Kernel)
    // La syntaxe <<< >>> est spécifique à CUDA. C'est le "coup de pistolet" du départ.
    // Le CPU envoie l'ordre et continue sa vie sans attendre (asynchrone).
    sepiaKernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height);

    // 4. Attente (Synchronisation)
    // Le CPU attend que le GPU ait fini avant de récupérer les résultats.
    cudaDeviceSynchronize();
}

```

### 4.4. Analyse de Performance

Est-ce que tout cet effort de programmation en vaut la peine ? Voici une comparaison typique pour le traitement d'une image haute définition.

| Métrique | CPU (Intel Core i7) | GPU (NVIDIA Tesla T4) | Gain |
| :--- | :--- | :--- | :--- |
| **Méthode** | Boucle séquentielle | 8 millions de threads parallèles | - |
| **Temps de calcul** | \~250 ms | \~3 ms | **x80** |
| **Philosophie** | Une Ferrari faisant 8 millions d'allers-retours | Un train de marchandises transportant tout d'un coup | - |

On peut remarquer que Les résultats sont sans appel : le GPU est infiniment plus efficace pour cette tâche. Mais pourquoi ?

**L'Explication Technique : Séquentiel vs Parallèle**

* **Le CPU** exécute une boucle `for` géante. Il doit traiter le pixel 1, *puis* le pixel 2, *puis* le 3... jusqu'au 8 300 000ème. Même s'il va très vite pour chaque pixel, l'addition des temps crée une latence élevée.

* **Le GPU** supprime la notion de temps pour la remplacer par de l'espace. Il n'attend pas que le pixel 1 soit fini pour commencer le 2. Il lance **tous les calculs en même temps** sur ses milliers d'unités de calcul.

> **🏎️ L'Analogie : La Livraison de Pizzas**
>
> Imaginez que vous devez livrer 8 millions de pizzas.
>
> * **Le CPU est une Ferrari.** Elle roule à 300 km/h. Mais elle ne peut transporter qu'une seule pizza à la fois. Elle doit faire 8 millions d'allers-retours.
> * **Le GPU est une armée de vélos (ou un train de marchandises).** Ils roulent lentement (20 km/h). Mais ils partent tous en même temps.
> **Résultat :** La première pizza livrée par la Ferrari arrive très vite (faible latence), mais pour livrer l'ensemble, l'armée de vélos finit des heures avant (haut débit).

### 4.5. Conclusion 

En résumé, il ne faut pas retenir que le GPU est "plus rapide" que le CPU (sa fréquence en MHz est souvent inférieure), mais qu'il est **massivement parallèle**.

Ce cours vous a invité à un changement de philosophie fondamental : nous sommes passés d'une architecture optimisée pour la **latence** (exécuter une tâche le plus vite possible) à une architecture dédiée au **débit** (exécuter des milliers de tâches simultanément).

Maîtriser CUDA, ce n'est pas seulement apprendre une nouvelle syntaxe. C'est comprendre comment transformer un problème temporel (attendre la fin d'une boucle) en un problème spatial (occuper toute la surface de la puce avec des milliers de threads). C'est cette capacité à "diviser pour régner" à grande échelle qui rend aujourd'hui possibles les avancées majeures en *Deep Learning* et en simulation scientifique.
