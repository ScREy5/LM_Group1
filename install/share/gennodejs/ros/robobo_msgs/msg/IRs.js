// Auto-generated. Do not edit!

// (in-package robobo_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let sensor_msgs = _finder('sensor_msgs');

//-----------------------------------------------------------

class IRs {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.FrontC = null;
      this.FrontR = null;
      this.FrontRR = null;
      this.FrontL = null;
      this.FrontLL = null;
      this.BackC = null;
      this.BackR = null;
      this.BackL = null;
    }
    else {
      if (initObj.hasOwnProperty('FrontC')) {
        this.FrontC = initObj.FrontC
      }
      else {
        this.FrontC = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('FrontR')) {
        this.FrontR = initObj.FrontR
      }
      else {
        this.FrontR = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('FrontRR')) {
        this.FrontRR = initObj.FrontRR
      }
      else {
        this.FrontRR = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('FrontL')) {
        this.FrontL = initObj.FrontL
      }
      else {
        this.FrontL = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('FrontLL')) {
        this.FrontLL = initObj.FrontLL
      }
      else {
        this.FrontLL = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('BackC')) {
        this.BackC = initObj.BackC
      }
      else {
        this.BackC = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('BackR')) {
        this.BackR = initObj.BackR
      }
      else {
        this.BackR = new sensor_msgs.msg.Range();
      }
      if (initObj.hasOwnProperty('BackL')) {
        this.BackL = initObj.BackL
      }
      else {
        this.BackL = new sensor_msgs.msg.Range();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type IRs
    // Serialize message field [FrontC]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.FrontC, buffer, bufferOffset);
    // Serialize message field [FrontR]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.FrontR, buffer, bufferOffset);
    // Serialize message field [FrontRR]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.FrontRR, buffer, bufferOffset);
    // Serialize message field [FrontL]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.FrontL, buffer, bufferOffset);
    // Serialize message field [FrontLL]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.FrontLL, buffer, bufferOffset);
    // Serialize message field [BackC]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.BackC, buffer, bufferOffset);
    // Serialize message field [BackR]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.BackR, buffer, bufferOffset);
    // Serialize message field [BackL]
    bufferOffset = sensor_msgs.msg.Range.serialize(obj.BackL, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type IRs
    let len;
    let data = new IRs(null);
    // Deserialize message field [FrontC]
    data.FrontC = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [FrontR]
    data.FrontR = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [FrontRR]
    data.FrontRR = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [FrontL]
    data.FrontL = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [FrontLL]
    data.FrontLL = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [BackC]
    data.BackC = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [BackR]
    data.BackR = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    // Deserialize message field [BackL]
    data.BackL = sensor_msgs.msg.Range.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += sensor_msgs.msg.Range.getMessageSize(object.FrontC);
    length += sensor_msgs.msg.Range.getMessageSize(object.FrontR);
    length += sensor_msgs.msg.Range.getMessageSize(object.FrontRR);
    length += sensor_msgs.msg.Range.getMessageSize(object.FrontL);
    length += sensor_msgs.msg.Range.getMessageSize(object.FrontLL);
    length += sensor_msgs.msg.Range.getMessageSize(object.BackC);
    length += sensor_msgs.msg.Range.getMessageSize(object.BackR);
    length += sensor_msgs.msg.Range.getMessageSize(object.BackL);
    return length;
  }

  static datatype() {
    // Returns string type for a message object
    return 'robobo_msgs/IRs';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3cda0da1598fe1278d12814909cfb3d7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    sensor_msgs/Range FrontC
    sensor_msgs/Range FrontR
    sensor_msgs/Range FrontRR
    sensor_msgs/Range FrontL
    sensor_msgs/Range FrontLL
    sensor_msgs/Range BackC
    sensor_msgs/Range BackR
    sensor_msgs/Range BackL
    
    ================================================================================
    MSG: sensor_msgs/Range
    # Single range reading from an active ranger that emits energy and reports
    # one range reading that is valid along an arc at the distance measured. 
    # This message is  not appropriate for laser scanners. See the LaserScan
    # message if you are working with a laser scanner.
    
    # This message also can represent a fixed-distance (binary) ranger.  This
    # sensor will have min_range===max_range===distance of detection.
    # These sensors follow REP 117 and will output -Inf if the object is detected
    # and +Inf if the object is outside of the detection range.
    
    Header header           # timestamp in the header is the time the ranger
                            # returned the distance reading
    
    # Radiation type enums
    # If you want a value added to this list, send an email to the ros-users list
    uint8 ULTRASOUND=0
    uint8 INFRARED=1
    
    uint8 radiation_type    # the type of radiation used by the sensor
                            # (sound, IR, etc) [enum]
    
    float32 field_of_view   # the size of the arc that the distance reading is
                            # valid for [rad]
                            # the object causing the range reading may have
                            # been anywhere within -field_of_view/2 and
                            # field_of_view/2 at the measured range. 
                            # 0 angle corresponds to the x-axis of the sensor.
    
    float32 min_range       # minimum range value [m]
    float32 max_range       # maximum range value [m]
                            # Fixed distance rangers require min_range==max_range
    
    float32 range           # range data [m]
                            # (Note: values < range_min or > range_max
                            # should be discarded)
                            # Fixed distance rangers only output -Inf or +Inf.
                            # -Inf represents a detection within fixed distance.
                            # (Detection too close to the sensor to quantify)
                            # +Inf represents no detection within the fixed distance.
                            # (Object out of range)
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new IRs(null);
    if (msg.FrontC !== undefined) {
      resolved.FrontC = sensor_msgs.msg.Range.Resolve(msg.FrontC)
    }
    else {
      resolved.FrontC = new sensor_msgs.msg.Range()
    }

    if (msg.FrontR !== undefined) {
      resolved.FrontR = sensor_msgs.msg.Range.Resolve(msg.FrontR)
    }
    else {
      resolved.FrontR = new sensor_msgs.msg.Range()
    }

    if (msg.FrontRR !== undefined) {
      resolved.FrontRR = sensor_msgs.msg.Range.Resolve(msg.FrontRR)
    }
    else {
      resolved.FrontRR = new sensor_msgs.msg.Range()
    }

    if (msg.FrontL !== undefined) {
      resolved.FrontL = sensor_msgs.msg.Range.Resolve(msg.FrontL)
    }
    else {
      resolved.FrontL = new sensor_msgs.msg.Range()
    }

    if (msg.FrontLL !== undefined) {
      resolved.FrontLL = sensor_msgs.msg.Range.Resolve(msg.FrontLL)
    }
    else {
      resolved.FrontLL = new sensor_msgs.msg.Range()
    }

    if (msg.BackC !== undefined) {
      resolved.BackC = sensor_msgs.msg.Range.Resolve(msg.BackC)
    }
    else {
      resolved.BackC = new sensor_msgs.msg.Range()
    }

    if (msg.BackR !== undefined) {
      resolved.BackR = sensor_msgs.msg.Range.Resolve(msg.BackR)
    }
    else {
      resolved.BackR = new sensor_msgs.msg.Range()
    }

    if (msg.BackL !== undefined) {
      resolved.BackL = sensor_msgs.msg.Range.Resolve(msg.BackL)
    }
    else {
      resolved.BackL = new sensor_msgs.msg.Range()
    }

    return resolved;
    }
};

module.exports = IRs;
