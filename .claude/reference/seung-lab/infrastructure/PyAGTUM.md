# PyAGTUM

**GitHub:** https://github.com/seung-lab/PyAGTUM
**Language:** Python | **Stars:** 1

Control software for an Automated Gridtape Ultramicrotome (AGTUM) system. Synchronizes a Leica UC7 ultramicrotome with a modified RMC ATUM tape collector and syringe pump for automated serial sectioning onto Luxel Gridtape.

## Key Features
- Synchronized ultramicrotome cutting with Gridtape aperture placement
- Water level monitoring via camera feed analysis with automatic syringe pump control
- Pre/post-sectioning camera monitoring of Gridtape
- Zaber 3-axis stage control for ATUM positioning
- Skip-cut functionality for avoiding bad nanofilm sections
- PyQt5 GUI interfaces

## Components
- `PyAGTUM.py` - Main UI for cutting control and tape synchronization
- `LeicaCamWater.py` - Camera-based water level monitoring and pump control
- `GridtapeCameras.py` - Pre/post sectioning camera feeds
- `hardwareUI.py` - Zaber stage positioning (tkinter)

## Relevance to Connectomics
Controls the physical sectioning hardware that produces the serial thin sections imaged by TEM/SEM for connectomics reconstruction.
