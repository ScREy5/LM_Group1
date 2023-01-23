// Auto-generated. Do not edit!

// (in-package robobo_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Tap {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.x = null;
      this.y = null;
    }
    else {
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = new std_msgs.msg.Int8();
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = new std_msgs.msg.Int8();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Tap
    // Serialize message field [x]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = std_msgs.msg.Int8.serialize(obj.y, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Tap
    let len;
    let data = new Tap(null);
    // Deserialize message field [x]
    data.x = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = std_msgs.msg.Int8.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a message object
    return 'robobo_msgs/Tap';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c455a05a0b8c9dc9827ed72d23a4de17';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int8 x
    std_msgs/Int8 y
    
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
    const resolved = new Tap(null);
    if (msg.x !== undefined) {
      resolved.x = std_msgs.msg.Int8.Resolve(msg.x)
    }
    else {
      resolved.x = new std_msgs.msg.Int8()
    }

    if (msg.y !== undefined) {
      resolved.y = std_msgs.msg.Int8.Resolve(msg.y)
    }
    else {
      resolved.y = new std_msgs.msg.Int8()
    }

    return resolved;
    }
};

module.exports = Tap;
