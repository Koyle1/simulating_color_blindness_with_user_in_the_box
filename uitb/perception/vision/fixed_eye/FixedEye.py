import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from collections import deque

from ...base import BaseModule
from ....utils.rendering import Camera
#from ..enumeration import Deficiency, Trichromatic_view

from daltonlens import convert, simulate, generate


import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum, auto

#color blindness simulation
class Trichromatic_view(Enum):
    NONE = auto()
    ANOMALOUS_TRICHROMACY = auto()
    DICHROMATIC = auto()
    ACHROMATOPSIA = auto()
    BLUE_CONE_MONOCHROMANCY = auto()

class Deficiency(Enum):
    NONE = auto()
    PROTAN = auto()
    DEUTAN = auto()
    TRITAN = auto()

class FixedEye(BaseModule):

  def __init__(self, model, data, bm_model, resolution, pos, quat, body="worldbody", channels=None, buffer=None, fps=30,
               **kwargs):
    """
    A simple eye model using a fixed camera.

    Args:
      model: A MjModel object of the simulation
      data: A MjData object of the simulation
      bm_model: A biomechanical model class object inheriting from BaseBMModel
      resolution: Resolution in pixels [width, height]
      pos: Position of the camera [x, y, z]
      quat: Orientation of the camera as a quaternion [w, x, y, z]
      body (optional): Body to which the camera is attached, default is 'worldbody'
      channels (optional): Which channels to use; 0-2 refer to RGB, 3 is depth. Default value is None, which means that all channels are used (i.e. same as channels=[0,1,2,3])
      buffer (optional): Defines a buffer of given length (in seconds) that is utilized to include prior observations
      **kwargs (optional): Keyword args that may be used

    Raises:
      KeyError: If "rendering_context" is not given (included in kwargs)
    """
    super().__init__(model, data, bm_model, **kwargs)

    self._model = model
    self._data = data
    self._fps = fps
    self._frame_interval = 1.0 / fps
    self._last_render_time = 0.0
    self.save_obs = True

    # Probably already called
    mujoco.mj_forward(self._model, self._data)

    # Set camera specs
    if channels is None:
      channels = [0, 1, 2, 3]
    self._channels = channels
    self._resolution = resolution
    self._pos = pos
    self._quat = quat
    self._body = body
    

    # Get rendering context
    if "rendering_context" not in kwargs:
      raise KeyError("rendering_context must be defined")
    self._context = kwargs["rendering_context"]

    if "trichromatic_view" not in kwargs:
        self.trichromatic_view = Trichromatic_view.NONE
    else:
        match kwargs["trichromatic_view"]:
            case 0:
                self.trichromatic_view = Trichromatic_view.NONE
            case 1:
                self.trichromatic_view = Trichromatic_view.ANOMALOUS_TRICHROMACY
            case 2:
                self.trichromatic_view = Trichromatic_view.DICHROMATIC
            case 3:
                self.trichromatic_view = Trichromatic_view.ACHROMATOPSIA
            case 4:
                self.trichromatic_view = Trichromatic_view.BLUE_CONE_MONOCHROMANCY
            case _:
                self.trichromatic_view = Trichromatic_view.NONE

    if "deficiency" not in kwargs:
        self.deficiency = Deficiency.NONE
    else:
        match kwargs["deficiency"]:
            case 0:
                self.deficiency = Deficiency.NONE
            case 1:
                self.deficiency = Deficiency.PROTAN
            case 2:
                self.deficiency = Deficiency.DEUTAN
            case 3:
                self.deficiency = Deficiency.TRITAN
            case _:
                self.deficiency = Deficiency.NONE

    if "severity" not in kwargs:
        self.severity = 0.8
    else:
        self.severity = kwargs["severity"]
    

    # Initialise camera
    self.camera_fixed_eye = Camera(context=self._context, model=model, data=data,
                          resolution=resolution, rgb=True, depth=True, camera_id="fixed-eye")
    self._camera_active = True

    # Append all cameras to self._cameras to be able to display
    # their outputs in human-view/GUI mode (used by simulator.py)
    self._cameras.append(self.camera_fixed_eye)

    # Define a vision buffer for including previous visual observations
    self._buffer = None
    if buffer is not None:
      assert "dt" in kwargs, "dt must be defined in order to include prior observations"
      maxlen = 1 + int(buffer/kwargs["dt"])
      self._buffer = deque(maxlen=maxlen)

  @staticmethod
  def insert(simulation, **kwargs):

    assert "pos" in kwargs, "pos needs to be defined for this perception module"
    assert "quat" in kwargs, "quat needs to be defined for this perception module"

    # Get simulation root
    simulation_root = simulation.getroot()

    # Add assets
    simulation_root.find("asset").append(ET.Element("mesh", name="eye", scale="0.05 0.05 0.05",
                                              file="assets/basic_eye_2.stl"))
    simulation_root.find("asset").append(ET.Element("texture", name="blue-eye", type="cube", gridsize="3 4",
                                              gridlayout=".U..LFRB.D..",
                                              file="assets/blue_eye_texture_circle.png"))
    simulation_root.find("asset").append(ET.Element("material", name="blue-eye", texture="blue-eye", texuniform="true"))

    # Create eye
    eye = ET.Element("body", name="fixed-eye", pos=kwargs["pos"], quat=kwargs["quat"])
    eye.append(ET.Element("geom", name="fixed-eye", type="mesh", mesh="eye", euler="0.69 1.43 0",
                          material="blue-eye", size="0.025", rgba="1.0 1.0 1.0 1.0"))
    eye.append(ET.Element("camera", name="fixed-eye", fovy="90"))

    # Add eye to a body
    body = kwargs.get("body", "worldbody")
    if body == "worldbody":
      simulation_root.find("worldbody").append(eye)
    else:
      eye_body = simulation_root.find(f".//body[@name='{body}'")
      assert eye_body is not None, f"Body with name {body} was not found"
      eye_body.append(eye)

  def get_observation(self, model, data, info=None):

    # Get rgb and depth arrays
    rgb, depth = self.camera_fixed_eye.render()
    assert not np.all(rgb==0), "There's still something wrong with rendering"

    
    # Normalise
    depth = (depth - 0.5) * 2

    # simulate color blindness
    rgb = self.simulate_colour_blindness(rgb = rgb, t = self.trichromatic_view, d = self.deficiency, severity = self.severity)

    if self.save_obs: 
        save_rgb = np.clip(rgb / 255.0, 0.0, 1.0)  # Convert to [0, 1] range
        filename = "obs_image.png"
        plt.imsave(filename, save_rgb)
        print(f"✅ Saved observation image to: {os.path.abspath(filename)}")
        self.save_obs = False

    rgb = (rgb / 255.0 - 0.5) * 2

    

    # Transpose channels
    obs = np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1]) 

    # Choose channels
    obs = obs[self._channels, :, :]

    

    # Include prior observation if needed
    if self._buffer is not None:
      # Update buffer
      if len(self._buffer) > 0:
        self._buffer.pop()
      while len(self._buffer) < self._buffer.maxlen:
        self._buffer.appendleft(obs)

      # Use latest and oldest observation, and their difference
      obs = np.concatenate([self._buffer[0], self._buffer[-1], self._buffer[-1] - self._buffer[0]], axis=0)

    return obs

  @property
  def camera_active(self):
    return self._camera_active

  @property
  def _default_encoder(self):
    return {"module": "rl.encoders", "cls": "SmallCNN", "kwargs": {"out_features": 256}}

  def _reset(self, model, data):
    if self._buffer is not None:
      self._buffer.clear()

  # @property
  # def encoder(self):
  #   return small_cnn(observation_shape=self._observation_shape, out_features=256)

  # Should perhaps create a base class for vision modules and define an abstract render function
  def render(self):
    current_time = time.time()
    if current_time - self._last_render_time >= self._frame_interval:
        self._last_render_time = current_time
        return self.camera_fixed_eye.render()
    else:
        # If not enough time elapsed, return None or last image if you want
        return None

  @staticmethod
  def simulate_colour_blindness(rgb, t: Trichromatic_view = Trichromatic_view.NONE, d: Deficiency = Deficiency.PROTAN, severity: float = 0.8, ):
      if t == Trichromatic_view.NONE:
          return rgb

      if t == Trichromatic_view.ANOMALOUS_TRICHROMACY:
          return FixedEye.simulate_anamolous_trichromatic_view(d, severity, rgb) # one channel does not work completely

      if t == Trichromatic_view.DICHROMATIC:
          return FixedEye.simulate_dichromatic(d, severity, rgb) # one channel does not work at all

      if t == Trichromatic_view.ACHROMATOPSIA:
          return FixedEye.simulate_archomatopsia(rgb) # grey sight

      if t == Trichromatic_view.BLUE_CONE_MONOCHROMANCY:
          return FixedEye.simulate_blue_cone_monochromancy(rgb) # blue sight

  @staticmethod
  def simulate_anamolous_trichromatic_view(d: Deficiency, severity:float, rgb):
      simulator = simulate.Simulator_Machado2009()
      
      # Map your Deficiency enum to daltonlens' Deficiency enum
      deficiency_map = {
          Deficiency.PROTAN: simulate.Deficiency.PROTAN,
          Deficiency.DEUTAN: simulate.Deficiency.DEUTAN,
          Deficiency.TRITAN: simulate.Deficiency.TRITAN,
      }
      
      dalton_d = deficiency_map[d]  # Translate your enum to what daltonlens expects
      
      altered_rgb = simulator.simulate_cvd(rgb, dalton_d, severity=severity)
      return altered_rgb

  @staticmethod
  def simulate_dichromatic(d: Deficiency, severity:float, rgb):
      simulator = simulate.Simulator_CoblisV2()
      
      # Map your Deficiency enum to daltonlens' Deficiency enum
      deficiency_map = {
          Deficiency.PROTAN: simulate.Deficiency.PROTAN,
          Deficiency.DEUTAN: simulate.Deficiency.DEUTAN,
          Deficiency.TRITAN: simulate.Deficiency.TRITAN,
      }
      
      dalton_d = deficiency_map[d]  # Translate your enum to what daltonlens expects

      altered_rgb = simulator.simulate_cvd(rgb, dalton_d, severity=severity)

      return altered_rgb

  @staticmethod
  def simulate_archomatopsia(rgb):
      gray_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
      gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

      return gray_3ch

  @staticmethod
  def simulate_blue_cone_monochromancy(rgb):
      # Convert RGB to BGR for OpenCV processing
      bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

      # Convert to LAB color space for better control over lightness and color
      lab_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab_image)

      # Desaturate the image by neutralizing A and B channels
      a[:] = 128
      b[:] = 128

      # Merge and convert back to BGR
      desaturated_image = cv2.merge((l, a, b))
      desaturated_bgr = cv2.cvtColor(desaturated_image, cv2.COLOR_LAB2BGR)

      # Emphasize the blue channel (channel 0 in BGR)
      desaturated_bgr[:, :, 0] = cv2.addWeighted(desaturated_bgr[:, :, 0], 1.5, np.zeros_like(desaturated_bgr[:, :, 0]), 0, 0)
 
      # Convert back to RGB before returning
      result_rgb = cv2.cvtColor(desaturated_bgr, cv2.COLOR_BGR2RGB)
    
      return result_rgb

    




    
    


