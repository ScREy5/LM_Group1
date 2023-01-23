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

class SetLedRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.id = null;
      this.color = null;
    }
    else {
      if (initObj.hasOwnProperty('id')) {
        this.id = initObj.id
      }
      else {
        this.id = new std_msgs.msg.String();
      }
      if (initObj.hasOwnProperty('color')) {
        this.color = initObj.color
      }
      else {
        this.color = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetLedRequest
    // Serialize message field [id]
    bufferOffset = std_msgs.msg.String.serialize(obj.id, buffer, bufferOffset);
    // Serialize message field [color]
    bufferOffset = std_msgs.msg.String.serialize(obj.color, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetLedRequest
    let len;
    let data = new SetLedRequest(null);
    // Deserialize message field [id]
    data.id = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    // Deserialize message field [color]
    data.color = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.id);
    length += std_msgs.msg.String.getMessageSize(object.color);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetLedRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '67de51c92b3fb96ec94dabeed286e658';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String id
    std_msgs/String color
    
    ================================================================================
    MSG: std_msgs/String
    string data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetLedRequest(null);
    if (msg.id !== undefined) {
      resolved.id = std_msgs.msg.String.Resolve(msg.id)
    }
    else {
      resolved.id = new std_msgs.msg.String()
    }

    if (msg.color !== undefined) {
      resolved.color = std_msgs.msg.String.Resolve(msg.color)
    }
    else {
      resolved.color = new std_msgs.msg.String()
    }

    return resolved;
    }
};

class SetLedResponse {
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
    // Serializes a message object of type SetLedResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetLedResponse
    let len;
    let data = new SetLedResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetLedResponse';
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
    const resolved = new SetLedResponse(null);
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
  Request: SetLedRequest,
  Response: SetLedResponse,
  md5sum() { return 'cae286d48c8a0ad77943140106a4ad56'; },
  datatype() { return 'robobo_msgs/SetLed'; }
};
