# MY NOTE

## 1. Separation in MNs:

- Definition:

  X & Y are separated in H give Z: 

  if there is no active trail in H between X & Y given Z

# 2. Factorization => Independece: MNs

- Theorem: If P factorizes over H, and $sep_H(X,Y \mid Z)$ then P satisfies $(X\perp Y \mid Z)$ 	

- $I(H)=\{(X\perp Y \mid Z): sep_H(X,Y\mid Z)\}$

  If **P** satisfies **I(H)**, we say that **H** is an **I-map** (independency map) of **P**

  **Theorem**: If **P** factorizes over **H**, then **H** is an **I-map** of **P**

# 3. Independence => Factorization:

- Theorem (Hammersley Clifford):

  For a positive distribution **P**, if **H** is an **I-map** for **P**, then **P** factorizes over **H**

# I-maps and perfect maps:

# 1. Capturing Independencies in P:

$I(P)=\{ (X\perp Y \mid Z): P \models (X \perp Y \mid Z) \}$

- P factorizes over G => G is an I-map for P
  -  $I(G) \subseteq I(P) $
  - But not always vice versa: there can be independencies in I(P) that are not in I(G)
- Want a Sparse Graph:
  - If the graph encodes more independencies
    - It is sparser (had fewer parameters) 
    - and more informative
  - Want a graph that captures as much of the structure in P as possible

# 2. Minimal I-map:

- Minimal I-map is I-map without redundant edges (I-map after remove all redundant edges)
- Minimal I-map may still not capture I(P)

![](C:\Users\robochat\Desktop\I_map not capture I_P.png)

# 3.  Perfect Map:

- Perfect map: $I(G) = I(P)$
  - G perfectly captures independencies in P

# 4. MN as a perfect map:

- Perfect map: I(H) = I(P)
  - H perfectly captures independencies in P 

# 5. Uniqueness of Perfect Map:

![](C:\Users\robochat\Desktop\Uniqueness of perfect map.png)

# 6. I-equivalence:

- Definition: 2 graphs $G_1$ and $G_2$ over $X_1, \ldots, X_n$ are I-equivalent if $I(G_1)=I(G_2)$

# 7. Summary:

- Graphs that capture more of I(P) are more compact and provide more insight 
- A minimal I-map ma fail to capture a lot of structure even if present and representable as a PGM
- A perfect map is great, but may not exist
- Converting BNs <-> MNs loses independencies
  - BN to MN: loses independencies in v-structures
  -  MN to BN: must add triangulating edges to loops

