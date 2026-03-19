# contact-points

**GitHub:** https://github.com/seung-lab/contact-points
**Language:** Python | **Stars:** 6

Algorithms for finding contact points between connected components in a 3D labeled image.

## Key Features
- Find contact points between any two labels in a 3D volume
- Returns coordinate pairs at contact boundaries
- Supports multi-label volumetric data

## API
```python
import contact_points
pts = contact_points.find_contact_points(data, label1, label2)
# returns [(x,y,z), (x,y,z), ...] alternating label1, label2
```

## Relevance to Connectomics
Identifies physical contact sites between neurons in segmented EM volumes, useful for synapse prediction and connectivity analysis.
