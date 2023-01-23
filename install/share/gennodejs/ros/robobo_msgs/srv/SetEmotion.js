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

class SetEmotionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.emotion = null;
    }
    else {
      if (initObj.hasOwnProperty('emotion')) {
        this.emotion = initObj.emotion
      }
      else {
        this.emotion = new std_msgs.msg.String();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetEmotionRequest
    // Serialize message field [emotion]
    bufferOffset = std_msgs.msg.String.serialize(obj.emotion, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetEmotionRequest
    let len;
    let data = new SetEmotionRequest(null);
    // Deserialize message field [emotion]
    data.emotion = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.emotion);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetEmotionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a6515dd33a2228f41cfa2f673f8eb2aa';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String emotion
    
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
    const resolved = new SetEmotionRequest(null);
    if (msg.emotion !== undefined) {
      resolved.emotion = std_msgs.msg.String.Resolve(msg.emotion)
    }
    else {
      resolved.emotion = new std_msgs.msg.String()
    }

    return resolved;
    }
};

class SetEmotionResponse {
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
    // Serializes a message object of type SetEmotionResponse
    // Serialize message field [error]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.error, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetEmotionResponse
    let len;
    let data = new SetEmotionResponse(null);
    // Deserialize message field [error]
    data.error = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'robobo_msgs/SetEmotionResponse';
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
    const resolved = new SetEmotionResponse(null);
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
  Request: SetEmotionRequest,
  Response: SetEmotionResponse,
  md5sum() { return 'b56a22e94c3e6cd89d04a799670be723'; },
  datatype() { return 'robobo_msgs/SetEmotion'; }
};
