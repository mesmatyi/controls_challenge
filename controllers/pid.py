from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.2
    self.i = 0.0
    self.d = 0.0
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      print("target_lataccel", target_lataccel)
      print("current_lataccel", current_lataccel)
      print("state", state)
      print("future_plan", future_plan)
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff
