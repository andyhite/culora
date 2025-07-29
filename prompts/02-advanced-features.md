# Culora Phase 2 Development Plan - Optimized for Stable Diffusion Training

## Phase 2 Overview

Building on the solid MVP foundation, Phase 2 focuses on **Stable Diffusion training optimization** through advanced analysis capabilities that directly improve training dataset quality. The goal is to implement the specific quality, composition, and content analysis needed to create optimal Stable Diffusion training datasets based on proven best practices from LAION and Stability AI.

## Strategic Objectives

1. **SD-Optimized Quality Assessment**: Multi-metric evaluation targeting high aesthetic scores and technical quality
2. **Composition-Aware Curation**: Shot type classification and framing analysis for balanced representation
3. **Content Safety & Cleanup**: Watermark detection, text overlay filtering, and content appropriateness
4. **Training-Ready Export**: Direct integration with popular SD training workflows
5. **Aesthetic Intelligence**: Visual appeal scoring and diversity optimization for high-quality outputs

---

## Epic 5: SD-Optimized Quality Assessment

### Research Tasks

- **RT5.1**: Benchmark aesthetic quality predictors (LAION aesthetic model, CLIP-based aesthetic scoring) against manual evaluation of SD training effectiveness
- **RT5.2**: Evaluate PIQ metrics (BRISQUE, SSIM, LPIPS) for correlation with SD training quality, focusing on detail preservation and artifact detection
- **RT5.3**: Research optimal resolution handling and upscaling detection for 1024px target training resolution

### User Stories

- **US5.1**: As a user, I want aesthetic quality scoring using LAION-style aesthetic predictors so my dataset contains visually appealing images that produce better SD outputs
- **US5.2**: As a user, I want enhanced technical quality assessment that detects compression artifacts, blur, and low-resolution images that hurt SD training
- **US5.3**: As a user, I want resolution validation that ensures images meet 1024px minimum requirements and flags upscaled/interpolated content
- **US5.4**: As a user, I want quality ensemble scoring that combines aesthetic appeal (60%) and technical quality (40%) optimized for SD training
- **US5.5**: As a user, I want quality distribution analysis showing aesthetic score ranges and recommendations for threshold tuning

---

## Epic 6: Composition Analysis for Balanced Representation

### Research Tasks

- **RT6.1**: Evaluate vision-language models for accurate shot type classification (portrait, medium shot, full body) and framing analysis
- **RT6.2**: Research optimal prompt strategies for detecting well-framed subjects vs. awkwardly cropped images
- **RT6.3**: Design composition diversity metrics that ensure balanced representation across angles, lighting, and contexts

### User Stories

- **US6.1**: As a user, I want accurate shot type classification with confidence scores so I can build balanced portrait/medium/full-body distributions
- **US6.2**: As a user, I want framing quality analysis that detects well-composed images vs. badly cropped or cut-off subjects
- **US6.3**: As a user, I want lighting and context diversity analysis (indoor/outdoor, natural/artificial) for comprehensive scene coverage
- **US6.4**: As a user, I want composition-based selection that ensures diverse angles, expressions, and backgrounds within each category
- **US6.5**: As a user, I want demographic and subject diversity tracking to prevent dataset bias and ensure balanced representation

---

## Epic 7: Content Safety and Dataset Cleanup

### Research Tasks

- **RT7.1**: Research watermark and text overlay detection methods optimized for training data cleanup
- **RT7.2**: Evaluate NSFW and content safety filtering approaches suitable for general-purpose SD training
- **RT7.3**: Design duplicate detection strategies that catch near-duplicates and repetitive content that could cause overfitting

### User Stories

- **US7.1**: As a user, I want automatic watermark and text overlay detection so I can exclude images that would teach unwanted artifacts to SD models
- **US7.2**: As a user, I want content safety filtering with configurable levels (strict/moderate/permissive) for different training goals
- **US7.3**: As a user, I want enhanced duplicate detection that catches near-duplicates and repetitive content to prevent overfitting
- **US7.4**: As a user, I want UI screenshot and text-heavy image filtering to focus the dataset on visual content rather than textual information
- **US7.5**: As a user, I want content category analysis that identifies and balances different subject types (people, objects, scenes, artwork)

---

## Epic 8: Training-Ready Export and Captioning

### Research Tasks

- **RT8.1**: Research SD training pipeline requirements (Kohya SS, EveryDream, Automatic1111) for optimal dataset organization
- **RT8.2**: Evaluate captioning strategies optimized for SD training (BLIP-2, CLIP Interrogator, manual templates)
- **RT8.3**: Design metadata export formats that preserve analysis results for training optimization

### User Stories

- **US8.1**: As a user, I want SD training-optimized captions that accurately describe visual content without irrelevant metadata
- **US8.2**: As a user, I want Kohya SS export format with proper directory structure, configuration files, and training-ready captions
- **US8.3**: As a user, I want EveryDream export format with JSON configuration optimized for the selected image characteristics
- **US8.4**: As a user, I want training metadata export including aesthetic scores, composition types, and quality metrics for training optimization
- **US8.5**: As a user, I want caption templates for different content types (portraits, objects, scenes, artwork) following SD training best practices

---

## Epic 9: Aesthetic Intelligence and Advanced Selection

### Research Tasks

- **RT9.1**: Integrate LAION aesthetic predictor or similar models for training-correlated aesthetic scoring
- **RT9.2**: Research diversity optimization strategies that maintain high aesthetic scores while ensuring content variety
- **RT9.3**: Design selection algorithms that balance aesthetic quality, technical quality, and compositional diversity

### User Stories

- **US9.1**: As a user, I want aesthetic score-based selection that prioritizes visually appealing images proven to improve SD training
- **US9.2**: As a user, I want diversity-aware aesthetic selection that maintains high visual appeal while ensuring content variety
- **US9.3**: As a user, I want selection strategies optimized for different SD training goals (general purpose, character-focused, artistic style)
- **US9.4**: As a user, I want aesthetic distribution analysis and recommendations for optimal training dataset composition
- **US9.5**: As a user, I want advanced filtering that combines aesthetic scores, technical quality, composition analysis, and content safety

---

## Implementation Sequence

### **Month 1: SD Quality Foundation**

- **Week 1-2**: Aesthetic quality assessment integration (LAION aesthetic predictor)
- **Week 3-4**: Enhanced technical quality with resolution validation and artifact detection

### **Month 2: Composition & Content Analysis**  

- **Week 5-6**: Shot type classification and framing quality analysis
- **Week 7-8**: Content safety filtering and watermark/text detection

### **Month 3: Training Integration**

- **Week 9-10**: SD training format exports (Kohya SS, EveryDream) with optimized captions
- **Week 11-12**: Training metadata and configuration generation

### **Month 4: Advanced Selection & Polish**

- **Week 13-14**: Aesthetic-aware diversity optimization and advanced selection algorithms
- **Week 15-16**: Performance optimization and comprehensive SD training workflow integration

## Technical Architecture for SD Optimization

### **New Dependencies**

```toml
# Aesthetic quality assessment
laion-aesthetic-predictor = "^1.0.0"
clip-by-openai = "^1.0"

# Enhanced quality assessment
piq = "^0.8.0"

# Vision-language for composition
transformers = "^4.36.0"
sentence-transformers = "^2.2.2"

# Content safety
nudenet = "^2.0.9"  # For NSFW detection
```

### **SD-Optimized Pipeline Architecture**

```txt
Input Images
    ↓
Stage 1: Resolution & Format Validation (1024px minimum, format check)
    ↓
Stage 2: Content Safety Filtering (NSFW, inappropriate content)
    ↓
Stage 3: Watermark & Text Overlay Detection (exclude training-harmful content)
    ↓
Stage 4: Technical Quality Assessment (blur, artifacts, compression)
    ↓
Stage 5: Aesthetic Quality Scoring (LAION-style aesthetic prediction)
    ↓
Stage 6: Composition Analysis (shot type, framing quality, diversity factors)
    ↓
Stage 7: Enhanced Duplicate Detection (prevent overfitting)
    ↓
SD-Optimized Selection (aesthetic + technical + composition + diversity)
    ↓
Training-Ready Export (captions + metadata + proper formatting)
```

### **SD-Specific Configuration**

```python
SD_OPTIMIZATION_CONFIG = {
    "quality_standards": {
        "min_resolution": 1024,
        "aesthetic_threshold": 5.0,  # LAION aesthetic score
        "technical_quality_min": 0.6,
        "blur_threshold": 100,  # Laplacian variance
        "compression_artifact_max": 0.3
    },
    "composition_targets": {
        "portrait": 0.4,           # Faces clearly visible
        "medium_shot": 0.35,       # Upper body, good for character training
        "full_body": 0.25,         # Complete subject, environmental context
        "framing_quality_min": 0.7 # Well-composed, not cut off
    },
    "content_safety": {
        "nsfw_threshold": 0.8,     # Configurable strictness
        "watermark_detection": True,
        "text_overlay_max": 0.1,   # Max text area ratio
        "ui_screenshot_filter": True
    },
    "diversity_optimization": {
        "max_aesthetic_sacrifice": 0.5,  # How much aesthetic score to sacrifice for diversity
        "demographic_balance": True,
        "lighting_variety": True,
        "background_diversity": True
    }
}
```

## Success Criteria for SD Training Optimization

### **Quality Metrics**

- Aesthetic scores align with manual evaluation of "good SD training images"
- 95%+ accuracy in watermark and text overlay detection
- Shot type classification accuracy >90% on diverse image sets
- Generated SD models show improved output quality on standard prompts

### **Training Effectiveness**

- Datasets produce SD models with better aesthetic consistency
- Reduced generation of unwanted artifacts (watermarks, text gibberish)
- Improved prompt adherence and compositional understanding
- Better demographic and stylistic diversity in generated outputs

### **User Experience**

- Clear reporting on why images were selected/rejected for SD training
- Training-ready exports work seamlessly with popular SD training tools
- Performance maintains <2 minutes per 100 images on modern hardware
- Configuration presets for different SD training scenarios (portraits, general, artistic)

This Phase 2 plan transforms Culora from a general dataset curator into a specialized Stable Diffusion training optimizer, implementing the specific quality and content analysis needed to build datasets that produce superior SD models based on proven training methodologies.
