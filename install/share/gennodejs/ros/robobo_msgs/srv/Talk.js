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

class TalkRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.text = null;
    }
    else {
      if (initObj.hasOwnProperty('text')) {
        this.text = initObj.text
      }
      else {
        this.text = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TalkRequest
    // Serialize message field [text]
    bufferOffset = std_msgs.msg.String.serialize(obj.text, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TalkRequest
    let len;
    let data = new TalkRequest(null);
    // Deserialize message field [text]
    data.text = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.text);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/TalkRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7fb91077b93c329c9eeaea1c488c03e0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String text
    
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
    const resolved = new TalkRequest(null);
    if (msg.text !== undefined) {
      resolved.text = std_msgs.msg.String.Resolve(msg.text)
    }
    else {
      resolved.text = new std_msgs.msg.String()
    }

    return resolved;
    }
};

class TalkResponse {
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
    // Serializes a message object of type TalkResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TalkResponse
    let len;
    let data = new TalkResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/TalkResponse';
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
    const resolved = new TalkResponse(null);
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
  Request: TalkRequest,
  Response: TalkResponse,
  md5sum() { return 'd8fdbf7e3675ec38cfc5f64bfa512f70'; },
  datatype() { return 'robobo_msgs/Talk'; }
};
