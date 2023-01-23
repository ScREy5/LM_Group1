// Auto-generated. Do not edit!

// (in-package robobo_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class MovePanTiltRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.panPos = null;
      this.panSpeed = null;
      this.panUnlockId = null;
      this.tiltPos = null;
      this.tiltSpeed = null;
      this.tiltUnlockId = null;
    }
    else {
      if (initObj.hasOwnProperty('panPos')) {
        this.panPos = initObj.panPos
      }
      else {
        this.panPos = new std_msgs.msg.Int16();
      }
      if (initObj.hasOwnProperty('panSpeed')) {
        this.panSpeed = initObj.panSpeed
      }
      else {
        this.panSpeed = new std_msgs.msg.Int8();
      }
      if (initObj.hasOwnProperty('panUnlockId')) {
        this.panUnlockId = initObj.panUnlockId
      }
      else {
        this.panUnlockId = new std_msgs.msg.Int16();
      }
      if (initObj.hasOwnProperty('tiltPos')) {
        this.tiltPos = initObj.tiltPos
      }
      else {
        this.tiltPos = new std_msgs.msg.Int16();
      }
      if (initObj.hasOwnProperty('tiltSpeed')) {
        this.tiltSpeed = initObj.tiltSpeed
      }
      else {
        this.tiltSpeed = new std_msgs.msg.Int8();
      }
      if (initObj.hasOwnProperty('tiltUnlockId')) {
        this.tiltUnlockId = initObj.tiltUnlockId
      }
      else {
        this.tiltUnlockId = new std_msgs.msg.Int16();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MovePanTiltRequest
    // Serialize message field [panPos]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.panPos, buffer, bufferOffset);
    // Serialize message field [panSpeed]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.panSpeed, buffer, bufferOffset);
    // Serialize message field [panUnlockId]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.panUnlockId, buffer, bufferOffset);
    // Serialize message field [tiltPos]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.tiltPos, buffer, bufferOffset);
    // Serialize message field [tiltSpeed]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.tiltSpeed, buffer, bufferOffset);
    // Serialize message field [tiltUnlockId]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.tiltUnlockId, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MovePanTiltRequest
    let len;
    let data = new MovePanTiltRequest(null);
    // Deserialize message field [panPos]
    data.panPos = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    // Deserialize message field [panSpeed]
    data.panSpeed = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    // Deserialize message field [panUnlockId]
    data.panUnlockId = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    // Deserialize message field [tiltPos]
    data.tiltPos = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    // Deserialize message field [tiltSpeed]
    data.tiltSpeed = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    // Deserialize message field [tiltUnlockId]
    data.tiltUnlockId = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 10;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/MovePanTiltRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e02b68971ce7a0ff6df448c278f194a2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int16 panPos
    std_msgs/Int8 panSpeed
    std_msgs/Int16 panUnlockId
    std_msgs/Int16 tiltPos
    std_msgs/Int8 tiltSpeed
    std_msgs/Int16 tiltUnlockId
    
    ================================================================================
    MSG: std_msgs/Int16
    int16 data
    
    ================================================================================
    MSG: std_msgs/Int8
    int8 data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MovePanTiltRequest(null);
    if (msg.panPos !== undefined) {
      resolved.panPos = std_msgs.msg.Int16.Resolve(msg.panPos)
    }
    else {
      resolved.panPos = new std_msgs.msg.Int16()
    }

    if (msg.panSpeed !== undefined) {
      resolved.panSpeed = std_msgs.msg.Int8.Resolve(msg.panSpeed)
    }
    else {
      resolved.panSpeed = new std_msgs.msg.Int8()
    }

    if (msg.panUnlockId !== undefined) {
      resolved.panUnlockId = std_msgs.msg.Int16.Resolve(msg.panUnlockId)
    }
    else {
      resolved.panUnlockId = new std_msgs.msg.Int16()
    }

    if (msg.tiltPos !== undefined) {
      resolved.tiltPos = std_msgs.msg.Int16.Resolve(msg.tiltPos)
    }
    else {
      resolved.tiltPos = new std_msgs.msg.Int16()
    }

    if (msg.tiltSpeed !== undefined) {
      resolved.tiltSpeed = std_msgs.msg.Int8.Resolve(msg.tiltSpeed)
    }
    else {
      resolved.tiltSpeed = new std_msgs.msg.Int8()
    }

    if (msg.tiltUnlockId !== undefined) {
      resolved.tiltUnlockId = std_msgs.msg.Int16.Resolve(msg.tiltUnlockId)
    }
    else {
      resolved.tiltUnlockId = new std_msgs.msg.Int16()
    }

    return resolved;
    }
};

class MovePanTiltResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.error = null;
    }
    else {
      if (initObj.hasOwnProperty('error')) {
        this.error = initObj.error
      }
      else {
        this.error = new std_msgs.msg.Int8();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MovePanTiltResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MovePanTiltResponse
    let len;
    let data = new MovePanTiltResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/MovePanTiltResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b41202f44ec8802b6a66ae15859258a4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8 error
    
    
    ================================================================================
    MSG: std_msgs/Int8
    int8 data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MovePanTiltResponse(null);
    if (msg.error !== undefined) {
      resolved.error = std_msgs.msg.Int8.Resolve(msg.error)
    }
    else {
      resolved.error = new std_msgs.msg.Int8()
    }

    return resolved;
    }
};

module.exports = {
  Request: MovePanTiltRequest,
  Response: MovePanTiltResponse,
  md5sum() { return 'cf92955e4ca6d280225977f6e00d71c8'; },
  datatype() { return 'robobo_msgs/MovePanTilt'; }
};
