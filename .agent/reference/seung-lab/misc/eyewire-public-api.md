# eyewire-public-api

**GitHub:** https://github.com/seung-lab/eyewire-public-api
**Language:** JavaScript | **Stars:** 8

Sample client application demonstrating the EyeWire REST API for citizen-science neuron reconstruction. Provides 2D/3D views of dataset and API usage examples including task assignment, validation submission, and OAuth2 authentication.

## Key Features
- REST API for assigning and submitting neuron tracing tasks
- Access to EM channel images and segmentation data
- OAuth2-based authentication for EyeWire accounts
- 3D mesh visualization of neuron reconstructions

## API
Key endpoints:
- `POST /1.0/tasks/assign` -- assign a tracing task
- `POST /1.0/tasks/:id/save` -- submit a validation

## Relevance to Connectomics
Public API for EyeWire, the crowd-sourced game for reconstructing neurons from EM data in the retina.
