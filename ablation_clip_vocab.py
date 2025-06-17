#!/usr/bin/env python3
"""
SGDGCN Spatial Activation Analysis - Data-Driven Approach
Instead of imposing predefined parts, this discovers what atoms actually learn spatially.
Modified to use class-specific concepts while allowing overlap between classes.
"""
from collections import Counter

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
# Import your model components
try:
    from attndictmodel import Net, create_adj, load_data
    from torch_geometric.data import Batch
except ImportError as e:
    print(f"Error importing model components: {e}")
    exit(1)


try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("CLIP not available - using simple spatial analysis")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def denormalize_image(img_tensor):
    """Denormalize image tensor"""
    device = img_tensor.device
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
    return torch.clamp(img_tensor * std + mean, 0, 1)

def get_class_specific_concepts(class_name):
    """Get concepts specific to each class with overlapping concepts allowed"""
    
    # Common concepts that can appear in multiple classes
    common_geometric = ["edge", "corner", "line", "curve", "texture", "pattern"]
    common_materials = ["metal", "glass", "plastic", "fabric", "skin", "surface"]
    common_anatomical = ["eye", "head", "leg", "body"]  # Can be shared across animals and humans
    common_environmental = ["grass", "sky", "water", "tree", "cloud", "road"]
    
    # Class-specific concept mappings
    class_concepts = {
        # Animals
        "cat": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "tail", "paw", "fur", "whisker", "feline", "animal"] + common_environmental,
        
        "dog": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "tail", "paw", "fur", "canine", "animal"] + common_environmental,
        
        "person": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "hand", "arm", "human", "clothing", "accessory"] + common_materials + common_environmental,
        
        "horse": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "tail", "mane", "hoof", "animal", "equine"] + common_environmental,
        
        "cow": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "tail", "udder", "horn", "animal", "bovine"] + common_environmental,
        
        "sheep": common_geometric + common_anatomical + ["nose", "ear", "mouth", "face", "tail", "wool", "animal", "ovine"] + common_environmental,
        
        "bird": common_geometric + ["eye", "head", "body", "wing", "tail", "beak", "feather", "animal", "avian"] + common_environmental,
        
        # Vehicles
        "bicycle": common_geometric + common_materials + ["wheel", "frame", "handlebar", "seat", "pedal", "chain", "spoke", "vehicle"] + common_environmental,
        
        "motorbike": common_geometric + common_materials + ["wheel", "engine", "handlebar", "seat", "headlight", "exhaust", "vehicle"] + common_environmental,
        
        "car": common_geometric + common_materials + ["wheel", "window", "door", "light", "headlight", "bumper", "mirror", "vehicle"] + common_environmental,
        
        "bus": common_geometric + common_materials + ["wheel", "window", "door", "light", "vehicle", "large"] + common_environmental,
        
        "train": common_geometric + common_materials + ["wheel", "window", "door", "carriage", "locomotive", "track", "vehicle"] + common_environmental,
        
        "truck": common_geometric + common_materials + ["wheel", "window", "door", "light", "cargo", "vehicle", "large"] + common_environmental,
        
        "boat": common_geometric + common_materials + ["hull", "sail", "mast", "deck", "vehicle", "water", "marine"] + ["water", "ocean", "river"],
        
        "aeroplane": common_geometric + common_materials + ["wing", "engine", "window", "fuselage", "tail", "propeller", "vehicle", "aircraft"] + ["sky", "cloud"],
        
        # Objects
        "bottle": common_geometric + common_materials + ["neck", "cap", "label", "container", "cylindrical"],
        
        "chair": common_geometric + common_materials + ["seat", "backrest", "leg", "arm", "furniture"],
        
        "diningtable": common_geometric + common_materials + ["surface", "leg", "furniture", "rectangular"],
        
        "pottedplant": common_geometric + ["plant", "pot", "leaf", "stem", "flower", "soil", "container"] + common_environmental,
        
        "sofa": common_geometric + common_materials + ["seat", "cushion", "arm", "backrest", "furniture"],
        
        "tvmonitor": common_geometric + common_materials + ["screen", "frame", "display", "electronic", "device", "rectangular"]
    }
    

    if class_name.lower() in class_concepts:
        return class_concepts[class_name.lower()]
    else:
     
        return common_geometric + common_materials + common_anatomical + common_environmental + [
            "object", "thing", "shape", "form", "structure"
        ]

class SpatialAtomAnalyzer:
    """Analyze what atoms actually learn spatially without preconceptions"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
   
        if HAS_CLIP:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            print("✓ CLIP loaded for semantic analysis")
        else:
            self.clip_model = None
    
    def extract_spatial_activations(self, image):
        """Extract atom activations with full spatial information"""
        self.model.eval()
        
        with torch.no_grad():
            img = image.to(self.device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            
         
            Fet = self.model.extractor(img)
            W = create_adj(Fet, k=4, alpha=1).to(self.device)
            data_list = load_data(W, Fet)
            batch = Batch.from_data_list(data_list).to(self.device)
            
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            x_enc = self.model.node_encoder(x, edge_index, None)
            coeffs = self.model.dictionary_module(x_enc)
            

            if coeffs.shape[0] == 197:
                coeffs = coeffs[1:]
            
  
            n = int(np.sqrt(coeffs.shape[0]))
            if n * n != coeffs.shape[0]:
                n = 14
                coeffs = coeffs[:n*n]
            
            spatial_coeffs = coeffs.view(n, n, -1).cpu().numpy()
            
            return spatial_coeffs, n
    
    def find_atom_activation_regions(self, spatial_coeffs, threshold_percentile=75):
        """Find where each atom activates most strongly"""
        atom_regions = {}
        grid_size = spatial_coeffs.shape[0]
        
        for atom_idx in range(spatial_coeffs.shape[2]):
            atom_map = spatial_coeffs[:, :, atom_idx]
            
            # Find high activation regions
            threshold = np.percentile(atom_map, threshold_percentile)
            if threshold > 0:
                high_activation_mask = atom_map > threshold
                
                if high_activation_mask.sum() > 0:
                    # Find connected components of high activation
                    high_activation_uint8 = high_activation_mask.astype(np.uint8)
                    contours, _ = cv2.findContours(high_activation_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    regions = []
                    for contour in contours:
                        if cv2.contourArea(contour) >= 1:  # At least 1 patch
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Convert to image coordinates (224x224)
                            patch_size = 224 // grid_size
                            img_x1 = x * patch_size
                            img_y1 = y * patch_size
                            img_x2 = min(224, (x + w) * patch_size)
                            img_y2 = min(224, (y + h) * patch_size)
                            
                            # Calculate average activation in this region
                            region_activation = atom_map[y:y+h, x:x+w].mean()
                            
                            regions.append({
                                'bbox': (img_x1, img_y1, img_x2, img_y2),
                                'grid_bbox': (x, y, x+w, y+h),
                                'activation': region_activation,
                                'area': w * h
                            })
                    
                    if regions:
                        # Sort by activation strength
                        regions.sort(key=lambda r: r['activation'], reverse=True)
                        atom_regions[atom_idx] = regions
        
        return atom_regions
    
    def analyze_region_content(self, image, region_bbox, class_name):
        """Analyze what's actually in a spatial region using CLIP with class-specific concepts"""
        if self.clip_model is None:
            return "visual_pattern"
        
        try:
            # Extract region from image
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            x1, y1, x2, y2 = region_bbox
            
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(224, x2), min(224, y2)
            
            if x2 <= x1 or y2 <= y1:
                return "invalid_region"
            
            region = img_np[y1:y2, x1:x2]
            
            if region.size == 0:
                return "empty_region"
            
            # Convert to PIL for CLIP
            region_uint8 = (region * 255).astype(np.uint8)
            region_pil = Image.fromarray(region_uint8)
            
            # Resize for CLIP
            region_pil = region_pil.resize((224, 224))
            
            # Get class-specific concepts
            concepts = get_class_specific_concepts(class_name)
            
            # Process through CLIP
            region_tensor = self.clip_preprocess(region_pil).unsqueeze(0)
            text_tokens = clip.tokenize([f"a photo of {concept}" for concept in concepts])
            
            with torch.no_grad():
                region_features = self.clip_model.encode_image(region_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                region_features /= region_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarities = (region_features @ text_features.T).squeeze()
                
                # Get top 2 concepts
                top_indices = similarities.argsort(descending=True)[:2]
                top_concepts = [concepts[i] for i in top_indices]
                
                return ", ".join(top_concepts)
        
        except Exception as e:
            print(f"Error analyzing region: {e}")
            return "analysis_error"
    
    def create_spatial_discovery_visualization(self, test_loader, num_examples=6, save_path="spatial_discovery.png"):
        """Create visualization showing what atoms actually learn spatially"""
        
        examples = []
        processed_count = 0
        
        print(f"Discovering spatial patterns in {num_examples} examples...")
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if len(examples) >= num_examples:
                break
                
            for i in range(images.shape[0]):
                if len(examples) >= num_examples:
                    break
                    
                try:
                    image = images[i:i+1].to(self.device)
                    label = labels[i].item()
                    class_name = test_loader.dataset.classes[label]
                    
                    processed_count += 1
                    
                    # Get model prediction
                    output, _ = self.model(image)
                    if output.dim() == 1:
                        predicted_class = output.argmax().item()
                    else:
                        predicted_class = output.argmax(dim=-1).item()
                    
                    if predicted_class == label:  # Only correct predictions
                        # Extract spatial activations
                        spatial_coeffs, grid_size = self.extract_spatial_activations(image)
                        
                        # Find activation regions for each atom
                        atom_regions = self.find_atom_activation_regions(spatial_coeffs)
                        
                        if len(atom_regions) >= 3:  # Need at least 3 atoms with regions
                            # Analyze content of regions using class-specific concepts
                            img_for_analysis = denormalize_image(image.squeeze(0))
                            
                            atom_analyses = {}
                            for atom_idx, regions in atom_regions.items():
                                if regions:  # Take the strongest region
                                    region = regions[0]
                                    content = self.analyze_region_content(img_for_analysis, region['bbox'], class_name)
                                    atom_analyses[atom_idx] = {
                                        'region': region,
                                        'content': content
                                    }
                            
                            if len(atom_analyses) >= 3:
                                examples.append({
                                    'image': image,
                                    'class_name': class_name,
                                    'spatial_coeffs': spatial_coeffs,
                                    'atom_regions': atom_regions,
                                    'atom_analyses': atom_analyses,
                                    'grid_size': grid_size
                                })
                                print(f"  ✓ Example {len(examples)}: {class_name} "
                                      f"({len(atom_analyses)} atoms with regions)")
                    
                except Exception as e:
                    print(f"Error processing image {processed_count}: {e}")
                    continue
        
        if not examples:
            print("No suitable examples found!")
            return []
        
        # Create visualization
        print(f"\nCreating spatial discovery visualization...")
        fig = plt.figure(figsize=(28, 6 * len(examples)))
        gs = GridSpec(len(examples), 8, figure=fig, hspace=0.2, wspace=0.1)
        
        fig.suptitle('SGDGCN: Class-Specific Spatial Pattern Discovery', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        for ex_idx, example in enumerate(examples):
            self._create_spatial_example_visualization(example, fig, gs, ex_idx)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Spatial discovery visualization saved to {save_path}")
        return examples
    

    def _create_spatial_example_visualization(self, example, fig, gs, row_idx):
        """Create visualization for one spatial discovery example"""
        image = example['image']
        class_name = example['class_name']
        spatial_coeffs = example['spatial_coeffs']
        atom_analyses = example['atom_analyses']
        
        # Original image
        img_display = denormalize_image(image.squeeze(0)).cpu().numpy()
        img_display = np.transpose(img_display, (1, 2, 0))
        
        ax_orig = fig.add_subplot(gs[row_idx, 0])
        ax_orig.imshow(img_display)
        ax_orig.set_title(f'{class_name.upper()}\nClass-Specific Analysis\n{len(atom_analyses)} patterns',
                        fontweight='bold', fontsize=14)
        ax_orig.axis('off')
        
        # Sort atoms by activation strength
        sorted_atoms = sorted(atom_analyses.items(),
                            key=lambda x: x[1]['region']['activation'],
                            reverse=True)
        
        # Show top 7 atoms and their discovered spatial patterns
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'black']
        
        for i, (atom_idx, analysis) in enumerate(sorted_atoms[:7]):
            ax = fig.add_subplot(gs[row_idx, i+1])
            
            region = analysis['region']
            content = analysis['content']
            
            # Create heatmap
            atom_heatmap = spatial_coeffs[:, :, atom_idx]
            atom_heatmap_resized = cv2.resize(atom_heatmap, (224, 224))
            
            if atom_heatmap_resized.max() > 0:
                atom_heatmap_resized = atom_heatmap_resized / atom_heatmap_resized.max()
            
            # Create overlay
            overlay = img_display.copy()
            heatmap_colored = plt.cm.Reds(atom_heatmap_resized)[:, :, :3]
            overlay = 0.6 * overlay + 0.4 * heatmap_colored
            
            ax.imshow(overlay)
            
            # Highlight the discovered region
            x1, y1, x2, y2 = region['bbox']
            color = colors[i % len(colors)]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color=color, linewidth=4)
            ax.add_patch(rect)
            
            # Add region center
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            ax.plot(center_x, center_y, 'o', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=2)
            
            # Title with discovered content
            activation_str = f"α={region['activation']:.3f}"
            
            # Check for analysis errors and display a cleaner title
            if 'error' in content.lower():
                title_text = f"Atom {atom_idx}\nPattern\n{activation_str}"
            else:
                title_text = f"Atom {atom_idx}\n{content}\n{activation_str}"
                
            ax.set_title(title_text,
                        fontsize=14, fontweight='bold', color=color)
            ax.axis('off')
        
        # Add spatial pattern overlay on original image
        for i, (atom_idx, analysis) in enumerate(sorted_atoms[:7]):
            region = analysis['region']
            x1, y1, x2, y2 = region['bbox']
            color = colors[i % len(colors)]
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color=color, linewidth=2, alpha=0.7)
            ax_orig.add_patch(rect)
            
            # Add atom number label
            ax_orig.text(x1+2, y1+12, f'A{atom_idx}',
                        fontsize=14, color='white', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

class Args:
    def __init__(self, num_classes):
        self.alpha = 0.1
        self.kernels = 4
        self.num_features = 768
        self.nhid = 64
        self.num_heads = 4
        self.mean_num_nodes = 30
        self.num_classes = num_classes
        self.pooling_ratio = 0.5
        self.dropout_ratio = 0.5
        self.num_atoms = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes):
    args = Args(num_classes)
    model = Net(args).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    return model, args

def create_spatial_pattern_analysis(examples, save_path):
    """Analyze the spatial patterns discovered by atoms with class-specific concepts"""
    
    if not examples:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class-Specific Spatial Pattern Discovery Analysis', fontsize=18, fontweight='bold')
    
    # Collect all discovered patterns
    all_patterns = []
    pattern_counts = defaultdict(int)
    atom_pattern_map = defaultdict(list)
    class_pattern_map = defaultdict(lambda: defaultdict(int))
    
    for ex in examples:
        for atom_idx, analysis in ex['atom_analyses'].items():
            content = analysis['content']
            activation = analysis['region']['activation']
            class_name = ex['class_name']
            
            all_patterns.append((content, activation, class_name))
            pattern_counts[content] += 1
            atom_pattern_map[atom_idx].append(content)
            class_pattern_map[class_name][content] += 1
    
    # Plot 1: Most common discovered patterns across all classes
    ax1 = axes[0, 0]
    if pattern_counts:
        patterns = list(pattern_counts.keys())[:12]  # Top 12
        counts = [pattern_counts[p] for p in patterns]
        
        bars = ax1.bar(range(len(patterns)), counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(patterns))))
        ax1.set_xticks(range(len(patterns)))
        ax1.set_xticklabels([p.replace(', ', '\n') for p in patterns], 
                           rotation=45, ha='right', fontsize=12)
        ax1.set_ylabel('Discovery Frequency')
        ax1.set_title('Most Discovered Class-Specific Patterns')
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Activation strength distribution
    ax2 = axes[0, 1]
    activations = [p[1] for p in all_patterns]
    if activations:
        ax2.hist(activations, bins=20, color='lightblue', edgecolor='navy', alpha=0.7)
        ax2.axvline(np.mean(activations), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(activations):.3f}')
        ax2.set_xlabel('Activation Strength')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Spatial Activation Distribution')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Patterns by class
    ax3 = axes[1, 0]
    if class_pattern_map:
        classes = list(class_pattern_map.keys())
        pattern_diversity = [len(patterns) for patterns in class_pattern_map.values()]
        
        bars = ax3.bar(range(len(classes)), pattern_diversity,
                      color=plt.cm.Set2(np.linspace(0, 1, len(classes))))
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Unique Patterns Discovered')
        ax3.set_title('Pattern Diversity by Object Class')
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Atom specialization
    ax4 = axes[1, 1]
    atom_specialization = {}
    for atom_idx, patterns in atom_pattern_map.items():
        if patterns:
            pattern_counts_for_atom = Counter(patterns)
            total = len(patterns)
            specialization = max(pattern_counts_for_atom.values()) / total
            atom_specialization[atom_idx] = specialization
    
    if atom_specialization:
        sorted_atoms = sorted(atom_specialization.items(), key=lambda x: x[1], reverse=True)
        atoms, spec_scores = zip(*sorted_atoms[:15])  # Top 15
        
        bars = ax4.bar(range(len(atoms)), spec_scores,
                      color=plt.cm.viridis(np.linspace(0, 1, len(atoms))))
        ax4.set_xticks(range(len(atoms)))
        ax4.set_xticklabels([f'A{a}' for a in atoms], fontsize=10)
        ax4.set_ylabel('Specialization Score')
        ax4.set_title('Atom Specialization (Top 15)')
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Class-specific spatial pattern analysis saved to: {save_path}")
    
    # Print detailed analysis
    print("\n" + "="*70)
    print("CLASS-SPECIFIC SPATIAL PATTERN DISCOVERY RESULTS")
    print("="*70)
    
    print(f"\nDiscovered Patterns (Top 10):")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pattern}: {count} discoveries")
    
    print(f"\nClass-Specific Pattern Examples:")
    for class_name, patterns in list(class_pattern_map.items())[:5]:
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        pattern_str = ", ".join([f"{p}({c})" for p, c in top_patterns])
        print(f"  {class_name}: {pattern_str}")
    
    if atom_specialization:
        print(f"\nMost Specialized Atoms:")
        for atom_idx, spec_score in sorted(atom_specialization.items(), key=lambda x: x[1], reverse=True)[:8]:
            main_pattern = max(set(atom_pattern_map[atom_idx]), key=atom_pattern_map[atom_idx].count)
            print(f"  Atom {atom_idx}: {spec_score:.3f} (mainly {main_pattern})")
    
    print(f"\nSummary:")
    print(f"  Total spatial regions analyzed: {len(all_patterns)}")
    print(f"  Unique patterns discovered: {len(pattern_counts)}")
    print(f"  Average activation strength: {np.mean(activations):.3f}")
    print(f"  Classes analyzed: {len(class_pattern_map)}")

def main():
    # Setup
    data_dir = '/home/annatar/projects/subgraph_dictionary_graph_behchmark_local/pascal'
    model_path = 'best_model_pascalvoc_trainval_loss.pth'
    output_dir = './class_specific_spatial_discovery'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'), 
        transform=data_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print(f"Dataset loaded: {len(test_dataset)} test images")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    print("Loading model...")
    num_classes = len(test_dataset.classes)
    model, args = load_model(model_path, num_classes)
    print(f"Model loaded successfully")
    
    # Create spatial analyzer
    print("Setting up class-specific spatial pattern analyzer...")
    analyzer = SpatialAtomAnalyzer(model)
    
    # Discover spatial patterns
    print("Discovering what atoms learn with class-specific concepts...")
    examples = analyzer.create_spatial_discovery_visualization(
        test_loader, 
        num_examples=8,
        save_path=os.path.join(output_dir, 'class_specific_spatial_discovery.png')
    )
    
    if examples:
        # Create analysis
        create_spatial_pattern_analysis(examples, os.path.join(output_dir, 'class_specific_spatial_analysis.png'))
        
        print("✓ Class-specific spatial discovery complete!")
        print(f"Generated files in {output_dir}:")
        print("  • class_specific_spatial_discovery.png - Class-specific atom patterns")
        print("  • class_specific_spatial_analysis.png - Pattern analysis")
        
        # Print class-specific concept examples
        print(f"\nClass-Specific Concepts Example:")
        for class_name in test_dataset.classes[:5]:
            concepts = get_class_specific_concepts(class_name)[:10]  # Show first 10
            print(f"  {class_name}: {', '.join(concepts)}")
        
    else:
        print("No examples found for spatial analysis")

if __name__ == '__main__':
    main()
