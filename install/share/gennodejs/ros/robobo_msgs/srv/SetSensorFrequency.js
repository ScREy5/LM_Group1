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

class SetSensorFrequencyRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.frequency = null;
    }
    else {
      if (initObj.hasOwnProperty('frequency')) {
        this.frequency = initObj.frequency
      }
      else {
        this.frequency = new std_msgs.msg.Int8();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetSensorFrequencyRequest
    // Serialize message field [frequency]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.frequency, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSensorFrequencyRequest
    let len;
    let data = new SetSensorFrequencyRequest(null);
    // Deserialize message field [frequency]
    data.frequency = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetSensorFrequencyRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b96ba08af418a90a3d51c1440b29c5db';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8 frequency   #0 for LOW
                              #1 for NORMAL
                              #2 for HIGH
                              #3 for MAX
    
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
    const resolved = new SetSensorFrequencyRequest(null);
    if (msg.frequency !== undefined) {
      resolved.frequency = std_msgs.msg.Int8.Resolve(msg.frequency)
    }
    else {
      resolved.frequency = new std_msgs.msg.Int8()
    }

    return resolved;
    }
};

class SetSensorFrequencyResponse {
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
    // Serializes a message object of type SetSensorFrequencyResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSensorFrequencyResponse
    let len;
    let data = new SetSensorFrequencyResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetSensorFrequencyResponse';
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
    const resolved = new SetSensorFrequencyResponse(null);
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
  Request: SetSensorFrequencyRequest,
  Response: SetSensorFrequencyResponse,
  md5sum() { return 'ea3a75b016df1ca9629d8a0e272b0401'; },
  datatype() { return 'robobo_msgs/SetSensorFrequency'; }
};
