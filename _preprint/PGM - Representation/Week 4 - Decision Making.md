# 1. Maximum Expected Utility: (Framework)

## a. Simple Decision Making:

A simple decision making situation **D**:

- A set of possible actions $Val(A) = \{a^1,\ldots,a^k \}$  (Different drug use for patient)
- A set of state $Val(X)=\{x^1,\ldots,x^N\}$ 
- A distribution P(X|A) (Which state will come when take particular action )
- A utility function $U(X, A)$ (4 Evaluation)

## b. Expected Utility: (EU)

- $EU[D[a]]=\sum_{x}P(x\mid a)U(x,a)$ 

- => Want to choose action $a$ that maximizes the expected utility

  $a^* = argmax_aEU[D[a]]$ 

## c. Expected Utility with Information:

- $EU[D[\delta_A]] = \sum_{x,a}P_{\delta_A}(x,a) U(x,a)$  (CPD $P(x \mid a)$ become Join Probability Distribution over  $X \cup [A]$)

- => Want to choose the decision rule $δ_A$ that maximizes the expected utility

  $argmax_{δ_A}EU[D[\delta_A]]$ 

  $MEU(D) = max_{δ_A}EU[D[\delta_A]]$   (MEU: maximum EU)

## d. Finding MEU Decision Rules:





# 2. Utility Function:

# 3. Value of Perfect Information

- $VPI(A \mid X)$ is the value of observing $X$ before choosing an action at $A$ (Value of perfect information)
- $D$ = original influence diagram
- $D_{X\rightarrow A}$ = influence diagram with edge $X \rightarrow A$ 
- $VPI(A \mid X) := MEU(D_{X \rightarrow A} - MEU(D))$ 
- Theorem: 
  - $VPI(A \mid X) \geq 0$
  - $VPI(A \mid X) = 0$ if and only if the optimal decision rule for $D$ is still optimal for $D_{X \rightarrow A}$  