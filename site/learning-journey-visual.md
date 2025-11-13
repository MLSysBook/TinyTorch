# Visual Learning Journey

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">The TinyTorch Learning Journey</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Visual roadmap from tensors to transformers</p>
</div>

**Purpose**: Visualize the learning progression, module dependencies, and milestone achievements in TinyTorch.

---

## The Complete Learning Flow

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e3f2fd','primaryTextColor':'#1976d2','primaryBorderColor':'#2196f3','lineColor':'#2196f3','secondaryColor':'#fff3e0','tertiaryColor':'#f3e5f5'}}}%%

flowchart TB
    Start([Start: Setup Environment]) --> M01[Module 01: Tensor]

    subgraph Foundation["🏗️ Foundation Tier (Modules 01-07)"]
        M01 --> M02[Module 02: Activations]
        M02 --> M03[Module 03: Layers]
        M03 --> M04[Module 04: Losses]
        M04 --> M05[Module 05: Autograd]
        M05 --> M06[Module 06: Optimizers]
        M06 --> M07[Module 07: Training]
    end

    M07 --> MS01{{"🏆 M01: 1957 Perceptron"}}
    M07 --> MS02{{"🏆 M02: 1969 XOR"}}

    MS02 --> M08[Module 08: DataLoader]

    M08 --> MS03{{"🏆 M03: 1986 MLP<br/>95%+ MNIST"}}

    subgraph Architecture["🏛️ Architecture Tier (Modules 08-13)"]
        M08 --> M09[Module 09: Spatial/CNNs]
        M08 --> M10[Module 10: Tokenization]

        M09 --> MS04{{"🏆 M04: 1998 CNN<br/>75%+ CIFAR-10"}}

        M10 --> M11[Module 11: Embeddings]
        M11 --> M12[Module 12: Attention]
        M12 --> M13[Module 13: Transformers]
    end

    M13 --> MS05{{"🏆 M05: 2017 Transformers<br/>Text Generation"}}

    subgraph Optimization["⚡ Optimization Tier (Modules 14-20)"]
        MS05 --> M14[Module 14: Profiling]
        M14 --> M15[Module 15: Quantization]
        M14 --> M16[Module 16: Compression]
        M14 --> M17[Module 17: Memoization]
        M15 --> M18[Module 18: Acceleration]
        M16 --> M18
        M17 --> M18
        M18 --> M19[Module 19: Benchmarking]
        M19 --> M20[Module 20: Competition]
    end

    M20 --> MS06{{"🏆 M06: 2024 Torch Olympics<br/>Production System"}}
    MS06 --> Complete([🎓 Complete!<br/>ML Systems Engineer])

    style M01 fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style M05 fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style M07 fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style M09 fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style M13 fill:#f3e5f5,stroke:#9c27b0,stroke-width:3px
    style M20 fill:#fce4ec,stroke:#e91e63,stroke-width:3px
    style MS01 fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style MS02 fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style MS03 fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style MS04 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    style MS05 fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style MS06 fill:#ffccbc,stroke:#ff5722,stroke-width:3px
    style Complete fill:#b2dfdb,stroke:#009688,stroke-width:4px
```

**Legend:**
- 🟦 Blue: Foundation modules
- 🟧 Orange highlights: Critical modules (Autograd, Training)
- 🟪 Purple: Advanced architecture modules
- 🟩 Green: Milestone achievements
- 🟨 Yellow: North Star milestone (CIFAR-10)
- 🟥 Red: Capstone

---

## Module Dependencies

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f5e9','primaryTextColor':'#2e7d32','primaryBorderColor':'#4caf50'}}}%%

graph LR
    subgraph Core["Core Foundation"]
        T[01 Tensor] --> A[02 Activations]
        T --> L[03 Layers]
        T --> Lo[04 Losses]
        T --> D[08 DataLoader]
    end

    subgraph Training["Training Engine"]
        T -.enhances.-> AG[05 Autograd]
        AG --> O[06 Optimizers]
        L --> O
        O --> TR[07 Training]
        Lo --> TR
    end

    subgraph Vision["Computer Vision"]
        T --> S[09 Spatial]
        A --> S
        L --> S
        AG --> S
    end

    subgraph Language["NLP Pipeline"]
        T --> TK[10 Tokenization]
        TK --> E[11 Embeddings]
        T --> E
        E --> AT[12 Attention]
        L --> AT
        AG --> AT
        AT --> TF[13 Transformers]
        A --> TF
        E --> TF
    end

    subgraph Opt["Optimization"]
        P[14 Profiling] --> Q[15 Quantization]
        P --> C[16 Compression]
        P --> M[17 Memoization]
        Q --> AC[18 Acceleration]
        C --> AC
        M --> AC
        AC --> B[19 Benchmarking]
        B --> CP[20 Competition]
    end

    TR --> S
    TR --> TF
    S -.optimized by.-> Opt
    TF -.optimized by.-> Opt

    style T fill:#ffeb3b,stroke:#f57c00,stroke-width:4px
    style AG fill:#ff9800,stroke:#e65100,stroke-width:4px
    style TR fill:#ff9800,stroke:#e65100,stroke-width:4px
    style S fill:#9c27b0,stroke:#4a148c,stroke-width:3px
    style TF fill:#9c27b0,stroke:#4a148c,stroke-width:3px
    style CP fill:#f44336,stroke:#b71c1c,stroke-width:3px
```

**Key Dependencies:**
- **Tensor (Module 01)**: Foundation for everything - all modules depend on it
- **Autograd (Module 05)**: Enhances Tensor, enables all learning
- **Training (Module 07)**: Orchestrates the complete learning pipeline
- **Vision & Language**: Parallel tracks that converge at optimization

---

## Three-Tier Structure

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%

timeline
    title TinyTorch Three-Tier Learning Journey

    section 🏗️ Foundation
        Module 01 : Tensor
        Module 02 : Activations
        Module 03 : Layers
        Module 04 : Losses
        Module 05 : Autograd
        Module 06 : Optimizers
        Module 07 : Training

    section 🏛️ Architecture
        Module 08 : DataLoader
        Module 09 : Spatial (CNNs)
        Module 10 : Tokenization
        Module 11 : Embeddings
        Module 12 : Attention
        Module 13 : Transformers

    section ⚡ Optimization
        Module 14 : Profiling
        Module 15 : Quantization
        Module 16 : Compression
        Module 17 : Memoization
        Module 18 : Acceleration
        Module 19 : Benchmarking
        Module 20 : Competition
```

---

## Historical Milestones Timeline

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#fff9c4','primaryTextColor':'#f57c00','primaryBorderColor':'#fbc02d'}}}%%

gantt
    title ML History Recreation Journey
    dateFormat YYYY
    axisFormat %Y

    section Milestones
    M01 1957 Perceptron          :milestone, 1957, 0d
    M02 1969 XOR Solution         :milestone, 1969, 0d
    M03 1986 MLP Revival          :milestone, 1986, 0d
    M04 1998 CNN Revolution       :milestone, 1998, 0d
    M05 2017 Transformer Era      :milestone, 2017, 0d
    M06 2024 Systems Age          :milestone, 2024, 0d

    section Your Progress
    Foundation (M01-07)           :active, 1957, 1969
    Architecture (M08-13)         :1969, 2017
    Optimization (M14-20)         :2017, 2024
```

**Journey Through ML History**: As you complete modules, you unlock milestones that recreate 67 years of machine learning breakthroughs using YOUR implementations.

---

## Student Learning Paths

```{mermaid}
%%{init: {'theme':'base'}}%%

flowchart TD
    Start([Choose Your Path]) --> Decision{Learning Goal?}

    Decision -->|"Fast: Understand ML"| Fast["🚀 Fast Track<br/>(6-8 weeks)<br/>Modules 01-09"]
    Decision -->|"Deep: Build Everything"| Complete["🎯 Complete Builder<br/>(14-18 weeks)<br/>All 20 Modules"]
    Decision -->|"Focus: Specific Skills"| Focused["🔍 Focused Explorer<br/>(8-12 weeks)<br/>Choose Tiers"]

    Fast --> F1[Foundation<br/>01-07]
    F1 --> F2[DataLoader<br/>08]
    F2 --> F3[Spatial/CNNs<br/>09]
    F3 --> FResult["✅ Can build & train<br/>neural networks<br/>75%+ CIFAR-10"]

    Complete --> C1[Foundation<br/>01-07]
    C1 --> C2[Architecture<br/>08-13]
    C2 --> C3[Optimization<br/>14-20]
    C3 --> CResult["🏆 ML Systems<br/>Engineer<br/>Production-ready"]

    Focused --> Choice{Focus Area?}
    Choice -->|Vision| FV[Foundation +<br/>Spatial 09]
    Choice -->|Language| FL[Foundation +<br/>NLP 10-13]
    Choice -->|Production| FO[Foundation +<br/>Optimization 14-20]

    FV --> FVResult["✅ Computer<br/>Vision Expert"]
    FL --> FLResult["✅ NLP/LLM<br/>Specialist"]
    FO --> FOResult["✅ ML Optimization<br/>Engineer"]

    style Fast fill:#e3f2fd,stroke:#2196f3
    style Complete fill:#f3e5f5,stroke:#9c27b0
    style Focused fill:#fff3e0,stroke:#f57c00
    style FResult fill:#c8e6c9,stroke:#4caf50
    style CResult fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    style FVResult fill:#c8e6c9,stroke:#4caf50
    style FLResult fill:#c8e6c9,stroke:#4caf50
    style FOResult fill:#c8e6c9,stroke:#4caf50
```

---

## Capability Progression

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e1f5fe','primaryTextColor':'#01579b'}}}%%

graph TB
    subgraph L1["Level 1: Foundation"]
        C1["Can create tensors<br/>and perform operations"]
        C2["Can build neural<br/>network layers"]
        C3["Can implement<br/>backpropagation"]
        C1 --> C2 --> C3
    end

    subgraph L2["Level 2: Training"]
        C4["Can train networks<br/>on datasets"]
        C5["Can achieve 95%+<br/>on MNIST"]
        C3 --> C4 --> C5
    end

    subgraph L3["Level 3: Architectures"]
        C6["Can build CNNs<br/>for vision"]
        C7["Can build transformers<br/>for language"]
        C8["Can achieve 75%+<br/>on CIFAR-10"]
        C5 --> C6 --> C8
        C5 --> C7
    end

    subgraph L4["Level 4: Production"]
        C9["Can profile and<br/>optimize models"]
        C10["Can compress 4×<br/>and speedup 10×"]
        C11["Can deploy production<br/>ML systems"]
        C8 --> C9
        C7 --> C9
        C9 --> C10 --> C11
    end

    C11 --> Master["🎓 ML Systems<br/>Mastery"]

    style C1 fill:#e3f2fd
    style C3 fill:#fff3e0
    style C5 fill:#f3e5f5
    style C8 fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    style C11 fill:#ffccbc
    style Master fill:#c8e6c9,stroke:#4caf50,stroke-width:4px
```

**Each level builds concrete, measurable capabilities** - not just "completed a module" but "can build production CNNs achieving 75%+ accuracy."

---

## Workflow Cycle

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#f0f4c3'}}}%%

graph LR
    Edit["📝 Edit Modules<br/>modules/source/XX_name/"] --> Export["⚙️ Export to Package<br/>tito module complete XX"]
    Export --> Validate["✅ Validate with Milestones<br/>milestones/0X_*/script.py"]
    Validate --> Check{Tests Pass?}
    Check -->|Yes| Next["➡️ Next Module"]
    Check -->|No| Debug["🔍 Debug & Fix"]
    Debug --> Edit
    Next --> Edit

    Validate -.optional.-> Progress["📊 Track Progress<br/>tito checkpoint status"]

    style Edit fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style Export fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Validate fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style Next fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style Debug fill:#ffcdd2,stroke:#f44336,stroke-width:2px
    style Progress fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5
```

**The essential three-step cycle**: Edit → Export → Validate

**📖 See [Student Workflow](student-workflow.md)** for detailed workflow guide.

---

## Dataset Strategy

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8eaf6'}}}%%

flowchart TB
    Start([Start Learning]) --> Ship["📦 Shipped Datasets<br/>(~350 KB in repo)"]

    Ship --> TD["TinyDigits<br/>1,200 samples<br/>8×8 images<br/>310 KB"]
    Ship --> TT["TinyTalks<br/>350 Q&A pairs<br/>Character-level<br/>40 KB"]

    TD --> M03["Milestone 03<br/>MLP on TinyDigits<br/>⚡ Fast iteration"]
    TT --> M05["Milestone 05<br/>Transformers on TinyTalks<br/>⚡ Instant training"]

    M03 --> Scale{Scale Up?}
    M05 --> Scale

    Scale -->|Yes| Download["⬇️ Downloaded Datasets<br/>(Auto-download when needed)"]

    Download --> MNIST["MNIST<br/>70K samples<br/>28×28 images<br/>10 MB"]
    Download --> CIFAR["CIFAR-10<br/>60K samples<br/>32×32 RGB<br/>170 MB"]

    MNIST --> M03B["Milestone 03<br/>MLP on MNIST<br/>🎯 95%+ accuracy"]
    CIFAR --> M04["Milestone 04<br/>CNN on CIFAR-10<br/>🏆 75%+ accuracy"]

    style Ship fill:#c8e6c9,stroke:#4caf50,stroke-width:2px
    style TD fill:#e3f2fd,stroke:#2196f3
    style TT fill:#e3f2fd,stroke:#2196f3
    style Download fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style MNIST fill:#f3e5f5,stroke:#9c27b0
    style CIFAR fill:#fff9c4,stroke:#fbc02d,stroke-width:3px
    style M04 fill:#ffccbc,stroke:#ff5722,stroke-width:3px
```

**Strategy**: Start small (shipped datasets), iterate fast, then validate on benchmarks (downloaded datasets).

**📖 See [Datasets Guide](datasets.md)** for complete dataset documentation.

---

## Success Metrics

```{mermaid}
%%{init: {'theme':'base'}}%%

mindmap
  root((TinyTorch<br/>Success))
    Technical Skills
      Build tensors from scratch
      Implement autograd engine
      Train real neural networks
      Achieve 75%+ CIFAR-10
      Optimize for production

    Understanding
      Know how PyTorch works internally
      Understand gradient flow
      Debug ML issues from first principles
      Profile and optimize bottlenecks

    Career Impact
      ML Systems Engineer role-ready
      Can implement novel architectures
      Production deployment skills
      Portfolio project (capstone)

    Milestones Achieved
      6 historical ML breakthroughs
      Recreated 67 years of ML history
      95%+ MNIST accuracy
      75%+ CIFAR-10 accuracy
```

---

## Time Investment vs. Outcomes

```{mermaid}
%%{init: {'theme':'base'}}%%

quadrantChart
    title Learning Paths: Time vs. Depth
    x-axis "Time Investment (weeks)"
    y-axis "ML Systems Mastery"
    quadrant-1 "Complete Mastery"
    quadrant-2 "Deep Understanding"
    quadrant-3 "Quick Learning"
    quadrant-4 "Focused Skills"

    "Fast Track (6-8w)": [0.35, 0.5]
    "Focused Vision (8w)": [0.45, 0.6]
    "Focused NLP (10w)": [0.55, 0.65]
    "Complete Builder (14-18w)": [0.85, 0.95]
    "Foundation Only (4w)": [0.25, 0.35]
```

**Quadrants:**
- **Bottom-left (Quick Learning)**: Foundation tier - understand basics in 4 weeks
- **Top-left (Deep Understanding)**: Fast track - build & train networks in 6-8 weeks
- **Bottom-right (Focused Skills)**: Specialized paths - vision or NLP focus
- **Top-right (Complete Mastery)**: Full course - ML systems engineer in 14-18 weeks

---

## Module Difficulty Progression

```{mermaid}
%%{init: {'theme':'base'}}%%

%%{init: {'theme':'base', 'themeVariables': { 'xyChart': {'backgroundColor': 'transparent'}}}}%%
xychart-beta
    title "Difficulty Curve Across 20 Modules"
    x-axis [M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20]
    y-axis "Difficulty (1-5 stars)" 0 --> 5
    line [2, 2, 3, 3, 4, 4, 4, 3, 5, 4, 4, 5, 5, 4, 5, 5, 4, 4, 4, 5]
```

**Key observations:**
- **Gentle start**: Modules 01-02 are beginner-friendly
- **First challenge**: Module 05 (Autograd) - the critical breakthrough
- **Sustained difficulty**: Modules 09, 12, 13, 15-16 are advanced (⭐⭐⭐⭐⭐)
- **Capstone peak**: Module 20 integrates everything

---

## Ready to Start?

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Begin Your Visual Journey</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">These diagrams show the path - now walk it!</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Start Building →</a>
<a href="learning-progress.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Track Progress →</a>
</div>

---

## Related Pages

- **📖 [Introduction](intro.md)** - What is TinyTorch and why build from scratch
- **📖 [Student Workflow](student-workflow.md)** - The essential edit → export → validate cycle
- **📖 [Three-Tier Structure](chapters/00-introduction.md)** - Detailed tier breakdown
- **📖 [Milestones](chapters/milestones.md)** - Journey through ML history
- **📖 [FAQ](faq.md)** - Common questions answered
