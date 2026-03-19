# seuron

**GitHub:** https://github.com/seung-lab/seuron
**Language:** Python | **Stars:** 7

SEUnglab neuRON pipeline -- a system for managing distributed reconstruction of neurons from EM image stacks. Supports local Docker Compose deployment and Google Cloud autoscaling.

## Key Features
- End-to-end EM neuron reconstruction pipeline (inference, segmentation, meshing)
- Distributed task management with autoscaling worker pools
- Slack bot and JupyterLab frontends for interactive control
- Local deployment via Docker Compose
- Google Cloud deployment with automatic instance group management
- Segmentation summary and resource usage reporting

## API
```bash
# Local deployment
git clone https://github.com/seung-lab/seuron.git && cd seuron
./start_seuronbot.local
# Then use the JupyterLab interface for pipeline control
```

## Relevance to Connectomics
Production orchestration system for large-scale EM neuron reconstruction, managing the full pipeline from ConvNet inference through segmentation and mesh generation.
