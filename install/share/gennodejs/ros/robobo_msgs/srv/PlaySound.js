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

class PlaySoundRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.sound = null;
    }
    else {
      if (initObj.hasOwnProperty('sound')) {
        this.sound = initObj.sound
      }
      else {
        this.sound = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PlaySoundRequest
    // Serialize message field [sound]
    bufferOffset = std_msgs.msg.String.serialize(obj.sound, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PlaySoundRequest
    let len;
    let data = new PlaySoundRequest(null);
    // Deserialize message field [sound]
    data.sound = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.sound);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/PlaySoundRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0935a125722801704c8ff8170000156e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String sound
    
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
    const resolved = new PlaySoundRequest(null);
    if (msg.sound !== undefined) {
      resolved.sound = std_msgs.msg.String.Resolve(msg.sound)
    }
    else {
      resolved.sound = new std_msgs.msg.String()
    }

    return resolved;
    }
};

class PlaySoundResponse {
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
    // Serializes a message object of type PlaySoundResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PlaySoundResponse
    let len;
    let data = new PlaySoundResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/PlaySoundResponse';
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
    const resolved = new PlaySoundResponse(null);
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
  Request: PlaySoundRequest,
  Response: PlaySoundResponse,
  md5sum() { return 'c95eacdcbf61c615623911c64d3b591f'; },
  datatype() { return 'robobo_msgs/PlaySound'; }
};
