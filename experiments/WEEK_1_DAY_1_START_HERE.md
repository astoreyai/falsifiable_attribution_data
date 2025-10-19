# WEEK 1 - DAY 1: IMPLEMENT REAL GRAD-CAM

**Goal**: Replace placeholder Grad-CAM with actual gradient-based attribution  
**Time**: 6-8 hours  
**Output**: Working Grad-CAM that produces real heatmaps  

---

## STEP 1: Read Current Implementation (10 min)

```bash
cd /home/aaron/projects/xai
cat src/attributions/gradcam.py
```

**Current state**: Lines 68-71 return `np.random.rand()` - this must be replaced.

---

## STEP 2: Install Dependencies (5 min)

```bash
# Should already be installed, but verify:
pip list | grep -E "torch|numpy"
```

---

## STEP 3: Implement Forward/Backward Hooks (60 min)

Open `src/attributions/gradcam.py` and replace the `GradCAM` class:

```python
class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Adapted for face verification (metric learning):
    - Target is embedding similarity, not class probability
    - Computes gradients of cosine similarity w.r.t. activations
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.target_layer_name = target_layer
        self.device = device
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """
        Find target convolutional layer.
        If target_layer_name specified, find that layer.
        Otherwise, use last Conv2d layer.
        """
        if self.target_layer_name:
            # Named layer (e.g., "layer4.2.conv3")
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    return module
            raise ValueError(f"Layer {self.target_layer_name} not found")
        else:
            # Find last Conv2d layer
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is None:
                raise ValueError("No Conv2d layer found in model")
            return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            # Store activations (detach to avoid gradient accumulation)
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Store gradients
            self.gradients = grad_output[0].detach()
        
        target_layer = self._find_target_layer()
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)
        
        # Store handles for cleanup
        self.hooks = [handle_forward, handle_backward]
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_attribution(
        self,
        image: torch.Tensor,
        target_embedding: Optional[torch.Tensor] = None,
        baseline_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM attribution map.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_embedding: Reference embedding for similarity (512,)
                            If None, uses embedding L2 norm as target
            baseline_embedding: Baseline for comparison (optional)
        
        Returns:
            attribution_map: Spatial heatmap (H, W) indicating important regions
        """
        self.model.eval()
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass to get embedding
        embedding = self.model(image)
        
        # Compute target score for backprop
        if target_embedding is not None:
            # Use cosine similarity as target
            target_embedding = target_embedding.to(self.device)
            target_score = F.cosine_similarity(
                embedding.view(1, -1),
                target_embedding.view(1, -1),
                dim=1
            )
        else:
            # Use embedding magnitude as target
            target_score = embedding.norm(dim=-1)
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        target_score.backward()
        
        # Retrieve activations and gradients
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")
        
        # Global Average Pooling of gradients (importance weights)
        # Shape: (1, C, H, W) -> (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        # Shape: (1, C, H, W) * (1, C, 1, 1) -> (1, C, H, W) -> (1, H, W)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Upsample to input image size
        cam = F.interpolate(
            cam,
            size=(image.shape[2], image.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().detach().numpy()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()
```

---

## STEP 4: Test on Single Image (30 min)

Create test script: `test_gradcam.py`

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.attributions.gradcam import GradCAM
from torchvision import transforms
from PIL import Image

# Load InsightFace model
import insightface
from insightface.app import FaceAnalysis

# Initialize
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load test image
image_path = '/path/to/test/face/image.jpg'  # UPDATE THIS
image = Image.open(image_path).convert('RGB')

# Preprocess
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image_tensor = transform(image).unsqueeze(0)  # (1, 3, 112, 112)

# Get embedding using InsightFace
faces = app.get(np.array(image))
if len(faces) == 0:
    print("No face detected!")
    exit()

embedding = torch.from_numpy(faces[0].embedding).float()

# Create wrapper model for Grad-CAM
class InsightFaceWrapper(torch.nn.Module):
    def __init__(self, app):
        super().__init__()
        self.app = app
        # Extract the recognition model
        self.rec_model = app.models['recognition']
    
    def forward(self, x):
        # x: (1, 3, 112, 112) tensor
        # Convert to numpy, run model, return embedding
        x_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x_np = (x_np * 0.5 + 0.5) * 255  # Denormalize
        x_np = x_np.astype(np.uint8)
        
        # Get embedding (this is simplified - may need bbox)
        emb = self.rec_model.get_feat([x_np])[0]
        return torch.from_numpy(emb).to(x.device)

model = InsightFaceWrapper(app)
model.eval()

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer=None, device='cuda')

# Compute attribution
attribution_map = gradcam.compute_attribution(
    image_tensor.cuda(),
    target_embedding=embedding.cuda()
)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Attribution heatmap
im = axes[1].imshow(attribution_map, cmap='jet')
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

# Overlay
axes[2].imshow(image)
axes[2].imshow(attribution_map, cmap='jet', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('gradcam_test.png', dpi=150, bbox_inches='tight')
print("Saved gradcam_test.png")
```

**Expected result**: Heatmap should highlight facial regions (eyes, nose, mouth), not random noise.

---

## STEP 5: Verify Implementation (30 min)

**Sanity checks:**

1. **Activation shape**: Print `self.activations.shape` - should be (1, C, H, W) where H,W < 112
2. **Gradient shape**: Print `self.gradients.shape` - should match activations
3. **Heatmap values**: Check `attribution_map.min()` and `attribution_map.max()` - should be [0, 1]
4. **Reproducibility**: Run twice with same image - should get identical results
5. **Face coverage**: Heatmap should have high values on face, low on background

**Debugging common issues:**

- **"No gradients captured"**: Check that `image.requires_grad = True` is set
- **"Layer not found"**: Print `model.named_modules()` to see available layers
- **Heatmap is all zeros**: Check if ReLU is too aggressive, try without it
- **Heatmap is uniform**: Weights might be constant, check gradient computation

---

## STEP 6: Integration Test (60 min)

Replace the placeholder in experiment files:

**File: `run_experiment_6_1.py`** (around line 250)

REMOVE:
```python
# Placeholder: return random attribution map for now
attribution_map = np.random.rand(H, W).astype(np.float32)
```

REPLACE WITH:
```python
from src.attributions.gradcam import GradCAM

# Initialize Grad-CAM (once, outside loop)
gradcam = GradCAM(model, target_layer=None, device=device)

# For each pair
attribution_map = gradcam.compute_attribution(
    image_tensor,
    target_embedding=ref_embedding
)
```

---

## STEP 7: Run Mini-Experiment (60 min)

Test with n=10 pairs (not full 200):

```bash
cd /home/aaron/projects/xai/experiments

# Modify run_experiment_6_1.py temporarily
# Change line: n_pairs = 200 -> n_pairs = 10

python run_experiment_6_1.py --device cuda --seed 42
```

**Check outputs:**
- Attribution maps should vary across images
- Falsification rates should NOT be exactly 45.2% (that was hardcoded)
- Should complete in ~5-10 minutes

---

## STEP 8: Commit Progress (10 min)

```bash
git add src/attributions/gradcam.py
git add test_gradcam.py
git commit -m "Implement real Grad-CAM with forward/backward hooks

- Replace placeholder with actual gradient computation
- Add activation/gradient capture via hooks
- Adapt for metric learning (cosine similarity target)
- Tested on single image, produces reasonable heatmaps
- Ready for full experiment integration"
```

---

## DELIVERABLES FOR DAY 1

- [ ] `src/attributions/gradcam.py` - Real implementation (not random)
- [ ] `test_gradcam.py` - Standalone test script
- [ ] `gradcam_test.png` - Visual verification that it works
- [ ] Mini-experiment results (n=10) showing non-hardcoded FRs
- [ ] Git commit with working Grad-CAM

---

## TOMORROW (DAY 2): Finish Grad-CAM Edge Cases

- Handle batch processing (multiple images at once)
- Add error handling for edge cases
- Optimize for speed (avoid recomputing embeddings)
- Document parameters and return values

---

## TROUBLESHOOTING

### Issue: "RuntimeError: element 0 of tensors does not require grad"
**Fix**: Ensure `image.requires_grad = True` before forward pass

### Issue: "Hooks not capturing gradients"
**Fix**: Check model is in eval mode: `model.eval()`

### Issue: "Attribution map is all zeros"
**Fix**: Try using embedding norm instead of similarity:
```python
target_score = embedding.norm()  # Instead of cosine_similarity
```

### Issue: "Can't find Conv2d layer"
**Fix**: InsightFace uses ONNX model, may need different hook strategy:
```python
# For ONNX models, try getting intermediate outputs differently
# Or use PyTorch version of ArcFace
```

---

## NEXT STEPS (AFTER DAY 1 COMPLETE)

**Day 2**: Polish Grad-CAM, add batch support
**Day 3**: Fix Experiment 6.2 aggregation bug
**Day 4**: Implement SHAP
**Day 5**: Implement falsification testing

---

**YOU GOT THIS!** Start with Step 1 and work through sequentially. Don't skip ahead.

