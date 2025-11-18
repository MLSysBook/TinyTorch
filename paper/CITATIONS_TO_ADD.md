# TinyTorch: Citations to Add - Quick Reference

## CRITICAL FIXES (Must do before submission)

### 1. Fix Corrupted Bib Entries

**bruner1960process** - Currently wrong paper about "Narrative Approach and Mentalization"

Replace with:
```bibtex
@book{bruner1960process,
  author = {Bruner, Jerome S.},
  title = {The Process of Education},
  year = {1960},
  publisher = {Harvard University Press},
  address = {Cambridge, MA}
}
```

OR use the original scaffolding paper:
```bibtex
@article{wood1976role,
  author = {Wood, David and Bruner, Jerome S. and Ross, Gail},
  title = {The role of tutoring in problem solving},
  journal = {Journal of Child Psychology and Psychiatry},
  volume = {17},
  number = {2},
  pages = {89--100},
  year = {1976},
  doi = {10.1111/j.1469-7610.1976.tb00381.x}
}
```

**perkins1992transfer** - Currently wrong paper about "infant mortality"

Replace with:
```bibtex
@incollection{perkins1992transfer,
  author = {Perkins, David N. and Salomon, Gavriel},
  title = {Transfer of Learning},
  booktitle = {International Encyclopedia of Education},
  edition = {2nd},
  year = {1992},
  publisher = {Pergamon Press},
  address = {Oxford, England}
}
```

---

## CRITICAL ADDITIONS (Strongly recommended)

### 2. Systems Thinking Foundation

```bibtex
@book{meadows2008thinking,
  author = {Meadows, Donella H.},
  title = {Thinking in Systems: A Primer},
  year = {2008},
  publisher = {Chelsea Green Publishing},
  address = {White River Junction, VT}
}
```

**Where to cite:** When using "systems thinking" terminology - add to Related Work Section 3.2

**Suggested text addition:**
> "TinyTorch's incremental system construction develops systems thinking~\citep{meadows2008thinking}—the ability to understand how components interact within complex systems—through direct implementation rather than abstract instruction."

### 3. Compiler Course Model

```bibtex
@book{aho2006compilers,
  author = {Aho, Alfred V. and Lam, Monica S. and Sethi, Ravi and Ullman, Jeffrey D.},
  title = {Compilers: Principles, Techniques, and Tools},
  edition = {2nd},
  year = {2006},
  publisher = {Addison-Wesley},
  address = {Boston, MA}
}
```

**Where to cite:** When comparing to "compiler course model" - Introduction and Related Work

**Suggested text addition:**
> "The curriculum follows compiler course pedagogy~\citep{aho2006compilers}: students build a complete system module-by-module, experiencing how components integrate through direct implementation."

### 4. Operating Systems Pedagogy

```bibtex
@book{arpaci2018operating,
  author = {Arpaci-Dusseau, Remzi H. and Arpaci-Dusseau, Andrea C.},
  title = {Operating Systems: Three Easy Pieces},
  year = {2018},
  publisher = {Arpaci-Dusseau Books},
  url = {http://pages.cs.wisc.edu/~remzi/OSTEP/}
}
```

**Where to cite:** When discussing "build the whole stack" approach

**Suggested text addition in Related Work:**
> "TinyTorch's incremental system construction draws pedagogical inspiration from compiler courses~\citep{aho2006compilers} and operating systems courses~\citep{arpaci2018operating}, where students build complete systems to develop systems thinking~\citep{meadows2008thinking} through component integration."

---

## HIGH-VALUE ADDITIONS

### 5. Active Learning Foundation

```bibtex
@article{freeman2014active,
  author = {Freeman, Scott and Eddy, Sarah L. and McDonough, Miles and Smith, Michelle K. and Okoroafor, Nnadozie and Jordt, Hannah and Wenderoth, Mary Pat},
  title = {Active learning increases student performance in science, engineering, and mathematics},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {111},
  number = {23},
  pages = {8410--8415},
  year = {2014},
  doi = {10.1073/pnas.1319030111}
}
```

**Where to cite:** When justifying hands-on, build-from-scratch approach

**Suggested text addition:**
> "This constructionist, hands-on pedagogy aligns with meta-analytic evidence that active learning significantly improves student performance in STEM education compared to traditional lecture-based instruction~\citep{freeman2014active}."

### 6. Academic Workforce Citations (Replace Robert Half/Keller)

```bibtex
@article{brynjolfsson2017machine,
  author = {Brynjolfsson, Erik and Mitchell, Tom},
  title = {What can machine learning do? Workforce implications},
  journal = {Science},
  volume = {358},
  number = {6370},
  pages = {1530--1534},
  year = {2017},
  doi = {10.1126/science.aap8062}
}

@article{ransbotham2020expanding,
  author = {Ransbotham, Sam and Kiron, David and Gerbert, Philipp and Reeves, Martin},
  title = {Expanding AI's Impact with Organizational Learning},
  journal = {MIT Sloan Management Review},
  volume = {61},
  number = {4},
  year = {2020}
}
```

**Where to use:** Introduction, replace or supplement Robert Half/Keller citations

**Suggested reframe:**
> "Machine learning systems engineering requires tacit knowledge that resists automation~\citep{brynjolfsson2017machine}: understanding why Adam requires 2× optimizer state memory, when attention's O(N²) scaling becomes prohibitive, how to navigate accuracy-latency-memory tradeoffs. While workforce demand for ML systems skills has grown substantially~\citep{ransbotham2020expanding}, traditional ML education separates algorithms from systems..."

### 7. Progressive Disclosure HCI/Learning

```bibtex
@article{mayer2009multimedia,
  author = {Mayer, Richard E. and Moreno, Roxana},
  title = {Nine ways to reduce cognitive load in multimedia learning},
  journal = {Educational Psychologist},
  volume = {38},
  number = {1},
  pages = {43--52},
  year = {2003},
  doi = {10.1207/S15326985EP3801_6}
}
```

OR

```bibtex
@book{nielsen1993usability,
  author = {Nielsen, Jakob},
  title = {Usability Engineering},
  year = {1993},
  publisher = {Academic Press},
  address = {Boston, MA}
}
```

**Where to cite:** When discussing progressive disclosure pattern as contribution

**Suggested text addition:**
> "Progressive disclosure—revealing complexity incrementally to manage cognitive load~\citep{mayer2009multimedia}—addresses this through runtime feature activation..."

---

## CITE WHAT YOU ALREADY HAVE

### 8. Bloom's Taxonomy for Assessment

**Already in bib:** `thompson2008bloom`

**Where to cite:** Section 4.4 (NBGrader Assessment) or when discussing assessment levels

**Suggested text addition:**
> "NBGrader tests progress from lower-order skills (recall tensor operations) to higher-order skills (analyze gradient flow, evaluate optimization tradeoffs), following Bloom's taxonomy for computing education~\citep{thompson2008bloom}."

### 9. Vygotsky - Scaffolding/ZPD

**Already in bib:** `vygotsky1978mind`

**Where to cite:** Related Work section when discussing scaffolding, or when discussing progressive difficulty

**Suggested text addition:**
> "Module prerequisites create scaffolding~\citep{vygotsky1978mind}, ensuring students work within their zone of proximal development: challenging enough to require new learning, familiar enough to build on mastered concepts."

---

## RECOMMENDED SEARCH QUERIES

### For Recent ML Education Work (2023-2024)

**Google Scholar:**
- `"machine learning education" 2023 2024 SIGCSE`
- `"teaching machine learning" 2023 2024 ICER`
- `"ML systems" education 2023 2024`

**ACM Digital Library:**
- Search SIGCSE 2023, 2024 proceedings for: "machine learning" OR "deep learning"
- Search ICER 2023, 2024 proceedings for: "ML" OR "framework"

**arXiv:**
- `cat:cs.CY "machine learning education" 2023`

**Target:** Find 2-3 recent papers showing awareness of current ML education research

---

## OPTIONAL BUT VALUABLE

### 10. Notional Machines

```bibtex
@phdthesis{sorva2013notional,
  author = {Sorva, Juha},
  title = {Notional Machines and Introductory Programming Education},
  school = {Aalto University},
  year = {2013},
  type = {Doctoral dissertation}
}
```

**Where to cite:** When discussing "mental models of framework internals"

**Suggested addition:**
> "Students develop notional machines~\citep{sorva2013notional}—mental models of framework internals—through implementation: how tensors store gradients, how autograd builds computational graphs, how optimizers manage state."

### 11. JAX Discussion in Related Work

**Already have bradbury2018jax in bib but not discussed**

**Suggested addition to Related Work:**
> "\textbf{JAX}~\citep{bradbury2018jax} offers an alternative functional paradigm through composable transformations (\texttt{jax.grad}, \texttt{vmap}). While pedagogically valuable for understanding functional programming applied to ML, JAX assumes framework usage rather than framework construction, positioning it complementary to TinyTorch's build-from-scratch approach."

---

## PRIORITY ORDER

### Must Do (1-2 hours):
1. Fix bruner1960process (corrupted)
2. Fix perkins1992transfer (corrupted)
3. Add meadows2008thinking (systems thinking)
4. Add aho2006compilers (compiler course model)
5. Cite vygotsky1978mind (already in bib)
6. Cite thompson2008bloom (already in bib)

### Should Do (2-3 hours):
7. Search SIGCSE/ICER 2023-2024 for recent work
8. Add brynjolfsson2017machine + ransbotham2020expanding (replace industry citations)
9. Add freeman2014active (active learning)
10. Add arpaci2018operating (OS course pedagogy)

### Could Do (1-2 hours):
11. Add mayer2009multimedia or nielsen1993usability (progressive disclosure)
12. Add sorva2013notional (notional machines)
13. Add JAX discussion in related work

---

## BIBTEX FILE CLEANUP

**Remove these if not cited in final paper:**
- Any uncited technical papers (baydin2018automatic, etc.)
- Duplicate or superseded entries

**Keep these even if lightly cited:**
- Historical ML papers (rosenblatt1958perceptron, rumelhart1986learning, lecun1998gradient) - important for milestone validation
- Technical framework papers (pytorch04release, tensorflow20) - important for historical context

---

## WHERE TO ADD IN PAPER

### Introduction (Lines 174-180):
- Replace/supplement Robert Half/Keller with Brynjolfsson, Ransbotham
- Add compiler/OS course references when comparing to "compiler course model"

### Related Work - Educational Frameworks (Lines 377-388):
- Add JAX discussion
- Cite recent ML education work if found

### Related Work - Learning Theory (Lines 389-400):
- Cite Vygotsky when discussing scaffolding
- Add Mayer/Nielsen for progressive disclosure grounding
- Add Freeman for active learning

### Related Work - New Subsection (INSERT):
Add new subsection "Systems Pedagogy Foundations":
```latex
\subsection{Systems Pedagogy Foundations}

TinyTorch's incremental system construction draws pedagogical inspiration from compiler courses~\citep{aho2006compilers} and operating systems courses~\citep{arpaci2018operating}, where students build complete systems (compilers from lexer to code generator, OS kernels from process management to file systems) to develop systems thinking~\citep{meadows2008thinking} through component integration. This "build the whole stack" approach has proven effective for teaching complex systems concepts in CS education.
```

### Section 4.4 NBGrader (Around line 820):
- Cite thompson2008bloom when discussing assessment levels

---

**Total estimated time for all Must Do + Should Do items: 3-5 hours**

**Expected impact on citation quality: 6.5/10 → 8.5/10**
