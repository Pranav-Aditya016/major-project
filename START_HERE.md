# 🚀 START HERE

Welcome! Your Medical Image Diffusion Pipeline is **complete and ready to use**.

---

## ⏱️ What You Need to Know (5 minutes)

### Your Project
- **Goal**: Generate synthetic medical images guided by binary lesion masks
- **Model**: DDPM (Diffusion) with mask conditioning
- **Novel Contribution**: Block Consistency Loss (enforces mask adherence)
- **Status**: ✅ PRODUCTION READY

### What's Installed
1. **Python Code** (src/) - 6 modules, fully documented
2. **Jupyter Notebooks** (notebooks/) - 2 complete scripts
3. **Documentation** - 5 comprehensive guides
4. **Data** - Task02_Heart.tar & Task05_Prostate.tar (ready to extract)

---

## 🎯 Quick Start (Choose Your Path)

### Path A: I Want to Test Quickly (5 minutes)
```
1. Open: notebooks/01_train_baseline.ipynb
2. Run: All cells (uses synthetic data, no files needed)
3. Result: See full training + sampling work
```

### Path B: I Want to Train on Real Data (15 minutes)
```
1. Open: notebooks/00_data_prep.ipynb
2. Run: All cells (extracts Heart/Prostate datasets)
3. Open: notebooks/01_train_baseline.ipynb
4. Edit: Update image_dir and mask_dir paths
5. Run: Training on real data
```

### Path C: I Want to Customize Everything (30+ minutes)
```
1. Read: README.md (full documentation)
2. Read: QUICK_REFERENCE.md (code examples)
3. Edit: src/config.py (adjust hyperparameters)
4. Create: Your own training script
5. Run: Custom experiments
```

---

## 📚 Documentation Map

| Document | Read When | Time |
|----------|-----------|------|
| **This file** | First | 5 min |
| README.md | Before coding | 15 min |
| QUICK_REFERENCE.md | While coding | 5 min (lookup) |
| MODULE_REFERENCE.md | Need API details | 10 min |
| IMPLEMENTATION_SUMMARY.md | Want technical overview | 10 min |
| COMPLETION_CHECKLIST.md | Verify everything | 5 min |
| PROJECT_COMPLETE.md | Final reference | 5 min |

---

## 💻 What to Run First

### Option 1: Test Everything (Synthetic Data)
**File**: `notebooks/01_train_baseline.ipynb`

This notebook:
- ✅ Uses synthetic data (no files needed)
- ✅ Runs full training loop
- ✅ Tests image generation
- ✅ Takes ~5 minutes
- ✅ Verifies GPU/CUDA works

**Do this first to validate the setup.**

### Option 2: Extract Real Datasets
**File**: `notebooks/00_data_prep.ipynb`

This notebook:
- ✅ Extracts Task02_Heart.tar
- ✅ Extracts Task05_Prostate.tar
- ✅ Organizes into images/masks folders
- ✅ Converts 3D to 2D slices
- ✅ Takes ~2 minutes

**Do this after validating setup, before real training.**

---

## 🔑 5 Key Concepts

### 1. Mask Conditioning
```
Input: Binary mask (256×256) indicating where lesion should be
       ↓
Model: Generates image with lesion in that region only
       ↓
Output: Synthetic image with lesion constrained to mask
```

### 2. Block Consistency Loss (Novel)
```
Standard DDPM: L = MSE(predicted_noise, true_noise)

Enhanced: L = L_DDPM + w × L_consistency

Where L_consistency penalizes:
  - Lesions NOT appearing in mask region
  - False lesions appearing outside mask
```

### 3. Diffusion Process
```
Forward: x_0 (clean image) → add noise → x_1000 (pure noise)
Reverse: x_1000 (noise) → denoise → x_0 (synthetic image)
```

### 4. Training Loop
```
1. Load image + mask batch
2. Add noise at random timestep
3. Predict noise with mask conditioning
4. Compute loss (diffusion + consistency)
5. Update weights
6. Repeat
```

### 5. Sampling
```
1. Start with random noise
2. Use mask to guide denoising
3. Iteratively remove noise (1000 → 0 steps)
4. Get synthetic image with lesion in mask region
```

---

## 🛠️ Common Setups

### Setup 1: Testing (No GPU Needed)
```python
device = 'cpu'  # Slower but works
config.batch_size = 1
config.model_channels = 32  # Very small
```

### Setup 2: Small GPU (4GB)
```python
device = 'cuda'
config.batch_size = 2
config.model_channels = 64
```

### Setup 3: Medium GPU (8GB)
```python
device = 'cuda'
config.batch_size = 8
config.model_channels = 128  # Default
```

### Setup 4: Large GPU (16GB+)
```python
device = 'cuda'
config.batch_size = 16
config.model_channels = 256
```

---

## ⚠️ Before Running

### Check GPU (if using CUDA)
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Check Python Version
```python
import sys
print(f"Python: {sys.version}")  # Should be 3.8+
```

### Verify Packages
```python
import torch
import monai
import nibabel
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ MONAI {monai.__version__}")
```

If any are missing, they're installed in setup.ipynb.

---

## 📊 Expected Results

### After Running 01_train_baseline.ipynb (Synthetic Data)

**Training Output:**
```
Epoch 0 Train Metrics:
  total: 0.123456
  diffusion: 0.100000
  consistency: 0.023456

Epoch 0 Val Metrics:
  loss: 0.098765

✓ Training complete!
Best validation loss: 0.098765
✓ Image generation test passed!
```

**Generated Image Properties:**
```
Shape: (1, 256, 256)
Range: [-1.0, 1.0]  (or [0.0, 1.0] depending on normalization)
Lesion visible: Only inside mask region ✓
```

---

## 🎯 Your First Training Run

### Step 1: Open the notebook
```
File → Open → notebooks/01_train_baseline.ipynb
```

### Step 2: Run cells in order
- Cell 1: Imports
- Cell 2: Configuration
- Cell 3: Create dataset
- Cell 4: Initialize model
- Cell 5: Train
- Cell 6: View metrics
- Cell 7: Generate samples

### Step 3: Watch the output
- Should see training progress bar
- Loss should decrease
- After training, see generated image

### Step 4: Save results
- Checkpoints auto-saved to `models/checkpoints/`
- Best model saved to `models/checkpoints/best_model.pt`

---

## 🔧 If Something Doesn't Work

### Problem: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're in the project root directory
```python
import sys
sys.path.insert(0, '/path/to/Major Project/src')
```

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size or model size
```python
config.batch_size = 2  # was 4
config.model_channels = 64  # was 128
```

### Problem: Notebook runs very slowly
**Solution**: Use CPU for testing, GPU for real training
```python
device = 'cpu'  # For fast testing
# Or check if CUDA is actually being used
```

### Problem: "FileNotFoundError: data/raw/Task02_Heart.tar"
**Solution**: This is expected - run 00_data_prep.ipynb after synthetic test

### More Issues?
→ See QUICK_REFERENCE.md (Debugging section)

---

## 📈 Next Steps After Testing

1. **Extract Real Data**: Run 00_data_prep.ipynb
2. **Train on Heart**: Update paths in 01_train_baseline.ipynb
3. **Tune Hyperparameters**: Edit configs or code
4. **Evaluate Results**: Generate samples and measure metrics
5. **Ablation Study**: Compare with/without Block Consistency Loss
6. **Scale Up**: Try Prostate or multi-organ training

---

## 📖 Key Files to Know

### Python Code (src/)
- `dataset.py` - Load images and masks
- `diffusion_model.py` - DDPM + Block Consistency Loss
- `config.py` - Hyperparameter management
- `trainer.py` - Training loop
- `data_prep.py` - Extract datasets

### Notebooks
- `00_data_prep.ipynb` - Extract Task02_Heart.tar, Task05_Prostate.tar
- `01_train_baseline.ipynb` - Full training script

### Documentation
- `README.md` - Full guide (start here for details)
- `QUICK_REFERENCE.md` - Fast lookup
- `MODULE_REFERENCE.md` - API documentation

---

## 💡 Pro Tips

1. **Start Small**: Use synthetic data first
2. **Monitor Losses**: Watch training/validation loss
3. **Save Configs**: Save hyperparameters with checkpoints
4. **Test Early**: Verify setup works before long training
5. **Use Checkpoints**: Save best model and resume from it

---

## 🎓 Research Notes

### For Your Paper
- Block Consistency Loss is the novel contribution
- Compare results with/without (w=0 vs w=0.1+)
- Report all hyperparameters
- Show lesion location accuracy

### Key Metrics
1. Lesion Localization Accuracy (did it follow mask?)
2. Image Quality (FID, LPIPS)
3. Training Convergence Speed
4. GPU Memory Usage

---

## ✅ Final Checklist Before You Start

- [ ] Python 3.8+ installed
- [ ] PyTorch installed (with CUDA if using GPU)
- [ ] MONAI installed
- [ ] setup.ipynb has been run
- [ ] You're in the project directory
- [ ] You've read this file (START HERE)
- [ ] You're ready to run 01_train_baseline.ipynb

---

## 🚀 You're Ready!

Everything is set up and documented. 

**Next action**: Open `notebooks/01_train_baseline.ipynb` and run the first cell.

If you have questions along the way:
1. Check QUICK_REFERENCE.md for code examples
2. Check README.md for detailed explanations
3. Check inline code comments
4. Check MODULE_REFERENCE.md for API docs

---

## Questions You Might Have

**Q: How long does training take?**
A: Synthetic data (1 epoch) = ~1 minute. Real data = depends on dataset size.

**Q: Can I use CPU instead of GPU?**
A: Yes! Just set `device='cpu'`. Much slower but works.

**Q: What's the Block Consistency Loss?**
A: It's your novel contribution - ensures lesions appear only in mask regions.

**Q: Can I train on multiple GPUs?**
A: Not yet, but can be added. For now, single GPU training.

**Q: Where do results get saved?**
A: Checkpoints → `models/checkpoints/`, Metrics → in notebook

**Q: How do I resume from a checkpoint?**
A: `trainer.load_checkpoint('models/checkpoints/best_model.pt')`

**Q: Can I evaluate on test data?**
A: Yes - load checkpoint, generate samples, compute metrics

---

**Happy researching! 🎉**

Start with: **notebooks/01_train_baseline.ipynb**
