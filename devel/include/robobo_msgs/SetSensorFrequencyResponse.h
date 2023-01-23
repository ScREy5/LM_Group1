// Generated by gencpp from file robobo_msgs/SetSensorFrequencyResponse.msg
// DO NOT EDIT!


#ifndef ROBOBO_MSGS_MESSAGE_SETSENSORFREQUENCYRESPONSE_H
#define ROBOBO_MSGS_MESSAGE_SETSENSORFREQUENCYRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Int8.h>

namespace robobo_msgs
{
template <class ContainerAllocator>
struct SetSensorFrequencyResponse_
{
  typedef SetSensorFrequencyResponse_<ContainerAllocator> Type;

  SetSensorFrequencyResponse_()
    : error()  {
    }
  SetSensorFrequencyResponse_(const ContainerAllocator& _alloc)
    : error(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Int8_<ContainerAllocator>  _error_type;
  _error_type error;





  typedef boost::shared_ptr< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> const> ConstPtr;

}; // struct SetSensorFrequencyResponse_

typedef ::robobo_msgs::SetSensorFrequencyResponse_<std::allocator<void> > SetSensorFrequencyResponse;

typedef boost::shared_ptr< ::robobo_msgs::SetSensorFrequencyResponse > SetSensorFrequencyResponsePtr;
typedef boost::shared_ptr< ::robobo_msgs::SetSensorFrequencyResponse const> SetSensorFrequencyResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator1> & lhs, const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator2> & rhs)
{
  return lhs.error == rhs.error;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator1> & lhs, const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace robobo_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b41202f44ec8802b6a66ae15859258a4";
  }

  static const char* value(const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb41202f44ec8802bULL;
  static const uint64_t static_value2 = 0x6a66ae15859258a4ULL;
};

template<class ContainerAllocator>
struct DataType< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "robobo_msgs/SetSensorFrequencyResponse";
  }

  static const char* value(const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Int8 error\n"
"\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Int8\n"
"int8 data\n"
;
  }

  static const char* value(const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.error);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetSensorFrequencyResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::robobo_msgs::SetSensorFrequencyResponse_<ContainerAllocator>& v)
  {
    s << indent << "error: ";
    s << std::endl;
    Printer< ::std_msgs::Int8_<ContainerAllocator> >::stream(s, indent + "  ", v.error);
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROBOBO_MSGS_MESSAGE_SETSENSORFREQUENCYRESPONSE_H
