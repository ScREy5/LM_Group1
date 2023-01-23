
"use strict";

let MovePanTilt = require('./MovePanTilt.js')
let MoveWheels = require('./MoveWheels.js')
let PlaySound = require('./PlaySound.js')
let ResetWheels = require('./ResetWheels.js')
let SetCamera = require('./SetCamera.js')
let SetEmotion = require('./SetEmotion.js')
let SetLed = require('./SetLed.js')
let SetSensorFrequency = require('./SetSensorFrequency.js')
let Talk = require('./Talk.js')

module.exports = {
  MovePanTilt: MovePanTilt,
  MoveWheels: MoveWheels,
  PlaySound: PlaySound,
  ResetWheels: ResetWheels,
  SetCamera: SetCamera,
  SetEmotion: SetEmotion,
  SetLed: SetLed,
  SetSensorFrequency: SetSensorFrequency,
  Talk: Talk,
};
