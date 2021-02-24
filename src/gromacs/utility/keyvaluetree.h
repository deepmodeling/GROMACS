/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \libinternal \file
 * \brief
 * Declares a data structure for JSON-like structured key-value mapping.
 *
 * A tree is composed of nodes that can have different types:
 *  - _Value_ (gmx::KeyValueTreeValue) is a generic node that can
 *    represent either a scalar value of arbitrary type, or an object or
 *    an array.
 *  - _Array_ (gmx::KeyValueTreeArray) is a collection of any number of values
 *    (including zero).  The values can be of any type and different types
 *    can be mixed in the same array.
 *  - _Object_ (gmx::KeyValueTreeObject) is a collection of properties.
 *    Each property must have a unique key.  Order of properties is preserved,
 *    i.e., they can be iterated in the order they were added.
 *  - _Property_ (gmx::KeyValueTreeProperty) is an arbitrary type of value
 *    associated with a string key.
 * The root object of a tree is typically an object, but could also be an
 * array.  The data structure itself does not enforce any other constraints,
 * but the context in which it is used can limit the allowed scalar types or,
 * e.g., require arrays to have values of uniform type.  Also, several
 * operations defined for the structure (string output, comparison,
 * serialization, etc.) only work on a limited set of scalar types, or have
 * limitations with the types of trees they work on (in particular, arrays are
 * currently poorly supported).
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_KEYVALUETREE_H
#define GMX_UTILITY_KEYVALUETREE_H

#include <algorithm>
#include <any>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

class KeyValueTreeArray;
class KeyValueTreeObject;
class TextWriter;

/*! \libinternal \brief
 * Identifies an entry in a key-value tree.
 *
 * This class is mainly an internal utility within the key-value tree
 * implementation, but it is exposed on the API level where string-based
 * specification of a location in the tree is necessary.  Constructors are not
 * explicit to allow passing a simple string in contexts where a
 * KeyValueTreePath is expected.
 *
 * The string specifying a location should start with a `/`, followed by the
 * names of the properties separated by `/`.  For example, `/a/b/c` specifies
 * property `c` in an object that is the value of `b` in an object that is the
 * value of `a` at the root of the tree.
 * Currently, there is no support for specifying paths to values within arrays
 * (since none of the places where this is used implement array handling,
 * either).
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
class KeyValueTreePath
{
public:
    //! Creates an empty path (corresponds to the root object).
    KeyValueTreePath() = default;
    //! Creates a path from given string representation.
    KeyValueTreePath(const char* path);
    //! Creates a path from given string representation.
    KeyValueTreePath(const std::string& path);

    //! Adds another element to the path, making it a child of the old path.
    void append(const std::string& key) { path_.push_back(key); }
    //! Adds elements from another path to the path.
    void append(const KeyValueTreePath& other)
    {
        auto elements = other.elements();
        path_.insert(path_.end(), elements.begin(), elements.end());
    }
    //! Removes the last element in the path, making it the parent path.
    void pop_back() { return path_.pop_back(); }
    //! Removes and returns the last element in the path.
    std::string pop_last()
    {
        std::string result = std::move(path_.back());
        path_.pop_back();
        return result;
    }

    //! Whether the path is empty (pointing to the root object).
    bool empty() const { return path_.empty(); }
    //! Returns the number of elements (=nesting level) in the path.
    size_t size() const { return path_.size(); }
    //! Returns the i'th path element.
    const std::string& operator[](int i) const { return path_[i]; }
    //! Returns all the path elements.
    const std::vector<std::string>& elements() const { return path_; }

    //! Formats the path as a string for display.
    std::string toString() const;

private:
    std::vector<std::string> path_;
};

//! \cond libapi

//! Combines two paths as with KeyValueTreePath::append().
inline KeyValueTreePath operator+(const KeyValueTreePath& a, const KeyValueTreePath& b)
{
    KeyValueTreePath result(a);
    result.append(b);
    return result;
}

//! Combines an element to a path as with KeyValueTreePath::append().
inline KeyValueTreePath operator+(const KeyValueTreePath& a, const std::string& b)
{
    KeyValueTreePath result(a);
    result.append(b);
    return result;
}
//! \endcond

class KeyValueTreeValue
{
public:
    //! Returns whether the value is an array (KeyValueTreeArray).
    bool isArray() const;
    //! Returns whether the value is an object (KeyValueTreeObject).
    bool isObject() const;
    //! Returns whether the value is of a given type.
    template<typename T>
    bool isType() const
    {
        return type() == typeid(T);
    }
    //! Returns the type of the value.
    std::type_index type() const { return value_.type(); }

    KeyValueTreeArray*        asArray();
    KeyValueTreeObject*       asObject();
    const KeyValueTreeArray&  asArray() const;
    const KeyValueTreeObject& asObject() const;
    template<typename T>
    const T& cast() const
    {
        if (!value_.has_value())
        {
            GMX_THROW(
                    InternalError("Attempt to access KeyValueTreeValue object as KeyValueTreeArray "
                                  "when nothing is present"));
        }
        if (!isType<T>())
        {
            throw std::bad_any_cast();
        }
        return *std::any_cast<T>(&value_);
    }
    template<typename T>
    T* castAsPointer()
    {
        if (!value_.has_value())
        {
            GMX_THROW(
                    InternalError("Attempt to access KeyValueTreeValue object as KeyValueTreeArray "
                                  "when nothing is present"));
        }
        if (!isType<T>())
        {
            throw std::bad_any_cast();
        }
        return std::any_cast<T>(&value_);
    }

    //! Returns the raw std::any value (always possible).
    const std::any& asAny() const { return value_; }

private:
    explicit KeyValueTreeValue(std::any&& value) : value_(std::move(value)) {}

    std::any value_;

    friend class KeyValueTreeBuilder;
    friend class KeyValueTreeObjectBuilder;
    friend class KeyValueTreeValueBuilder;
};

class KeyValueTreeArray
{
public:
    //! Whether all elements of the array are objects.
    bool isObjectArray() const
    {
        return std::all_of(values_.begin(), values_.end(), std::mem_fn(&KeyValueTreeValue::isObject));
    }

    //! Returns the values in the array.
    const std::vector<KeyValueTreeValue>& values() const { return values_; }

private:
    std::vector<KeyValueTreeValue> values_;

    friend class KeyValueTreeArrayBuilderBase;
};

class KeyValueTreeProperty
{
public:
    const std::string&       key() const { return value_->first; }
    const KeyValueTreeValue& value() const { return value_->second; }

private:
    typedef std::map<std::string, KeyValueTreeValue>::const_iterator IteratorType;

    explicit KeyValueTreeProperty(IteratorType value) : value_(value) {}

    IteratorType value_;

    friend class KeyValueTreeObject;
    friend class KeyValueTreeObjectBuilder;
};

} // namespace gmx

// The KeyValueTreeObject copy constructor needs to work around a
// libstdc++ bug with std::is_constructible applied to std::any,
// because of some non-standard content in the implementation of
// std::any. The fix of
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90415 should solve
// this in libstdc++ 9.4 and 10.1 and later (note that 10.0 is a
// pre-release version). Note that e.g. libstdc++ is used by default
// by clang on linux, so the issue can affect builds by any compiler.
//
// Fortunately, we can work around the bug by declaring a template
// specialization that a gmx type that contains a std::any is
// constructible. One is generally not supposed to extend the std
// namespace, but this seems a suitable exception.
//
// The copy constructor of KeyValueTreeObject has a static_assert
// that checks whether this work-around is needed and missing.
//
// This work-around and the matching static_assert can be removed when
// gcc 10 or later is required.
#if defined _GLIBCXX_RELEASE && _GLIBCXX_RELEASE < 10

namespace std
{
//! Declare the necessary custom specialization of std::is_constructible
template<>
struct is_constructible<gmx::KeyValueTreeValue, const gmx::KeyValueTreeValue&&>
{
    //! Declare the type of \c value
    using value_type = bool;
    //! Declare that gmx::KeyValueTreeValue is constructible
    const static value_type value = true;
    using type                    = std::integral_constant<bool, value>;

               operator bool() { return value; }
    value_type operator()() { return value; }
};

//! Declare the necessary custom specialization of std::is_constructible
template<>
struct is_constructible<gmx::KeyValueTreeValue, gmx::KeyValueTreeValue&&>
{
    //! Declare the type of \c value
    using value_type = bool;
    //! Declare that gmx::KeyValueTreeValue is constructible
    const static value_type value = true;
    using type                    = std::integral_constant<bool, value>;

               operator bool() { return value; }
    value_type operator()() { return value; }
};

//! Declare the necessary custom specialization of std::is_constructible
template<>
struct is_constructible<gmx::KeyValueTreeValue, const gmx::KeyValueTreeValue&>
{
    //! Declare the type of \c value
    using value_type = bool;
    //! Declare that gmx::KeyValueTreeValue is constructible
    const static value_type value = true;
    using type                    = std::integral_constant<bool, value>;

               operator bool() { return value; }
    value_type operator()() { return value; }
};

/* Not needed
//! Declare the necessary custom specialization of std::is_copy_constructible
template<>
struct is_copy_constructible<gmx::KeyValueTreeValue>
{
    //! Declare the type of \c value
    using value_type = bool;
    //! Declare that gmx::KeyValueTreeValue is copy constructible
    const static value_type value = true;
    using type                    = std::integral_constant<bool, value>;

        operator bool() { return value; }
    value_type operator()() { return value; }
};
*/

/* Something I tried in desperation, doesn't do anything
template <>
struct __is_move_insertable<std::allocator<gmx::KeyValueTreeValue>>
{
    //! Declare the type of \c value
    using value_type = bool;
    //! Declare that gmx::KeyValueTreeValue is move insertable
    const static value_type value = true;
    using type                    = std::integral_constant<bool, value>;

        operator bool() { return value; }
    value_type operator()() { return value; }
};
*/

} // namespace std

#endif // _GLIBCXX_RELEASE

namespace gmx
{

class KeyValueTreeObject
{
public:
    KeyValueTreeObject() = default;
    //! Creates a deep copy of an object.
    KeyValueTreeObject(const KeyValueTreeObject& other)
    {
        for (const auto& value : other.values_)
        {
            // Assert that the work-around is effective when it is active.
            static_assert(std::is_copy_constructible_v<KeyValueTreeValue>,
                          "Can copy-construct KVTV");
            auto iter = valueMap_.insert(std::make_pair(value.key(), value.value())).first;
            values_.push_back(KeyValueTreeProperty(iter));
        }
    }
    //! Assigns a deep copy of an object.
    KeyValueTreeObject& operator=(const KeyValueTreeObject& other)
    {
        KeyValueTreeObject tmp(other);
        std::swap(tmp.valueMap_, valueMap_);
        std::swap(tmp.values_, values_);
        return *this;
    }
    //! Default move constructor.
    //NOLINTNEXTLINE(performance-noexcept-move-constructor) bug #38733
    KeyValueTreeObject(KeyValueTreeObject&&) = default;
    //! Default move assignment.
    KeyValueTreeObject& operator=(KeyValueTreeObject&&) = default;

    /*! \brief
     * Returns all properties in the object.
     *
     * The properties are in the order they were added to the object.
     */
    const std::vector<KeyValueTreeProperty>& properties() const { return values_; }

    //! Whether a property with given key exists.
    bool keyExists(const std::string& key) const { return valueMap_.find(key) != valueMap_.end(); }
    //! Returns value for a given key.
    const KeyValueTreeValue& operator[](const std::string& key) const
    {
        GMX_ASSERT(keyExists(key), "Accessing non-existent value");
        return valueMap_.at(key);
    }

    /*! \brief
     * Returns whether the given object shares any keys with `this`.
     */
    bool hasDistinctProperties(const KeyValueTreeObject& obj) const;

private:
    //! Keeps the properties by key.
    std::map<std::string, KeyValueTreeValue> valueMap_;
    //! Keeps the insertion order of properties.
    std::vector<KeyValueTreeProperty> values_;

    friend class KeyValueTreeObjectBuilder;
};

/********************************************************************
 * Inline functions that could not be declared within the classes
 */

inline bool KeyValueTreeValue::isArray() const
{
    return isType<KeyValueTreeArray>();
}
inline bool KeyValueTreeValue::isObject() const
{
    return isType<KeyValueTreeObject>();
}
inline const KeyValueTreeArray& KeyValueTreeValue::asArray() const
{
    return cast<KeyValueTreeArray>();
}
inline const KeyValueTreeObject& KeyValueTreeValue::asObject() const
{
    return cast<KeyValueTreeObject>();
}
inline KeyValueTreeArray* KeyValueTreeValue::asArray()
{
    return castAsPointer<KeyValueTreeArray>();
}
inline KeyValueTreeObject* KeyValueTreeValue::asObject()
{
    return castAsPointer<KeyValueTreeObject>();
}

//! \cond libapi
/*! \brief
 * Writes a human-readable representation of the tree with given writer.
 *
 * The output format is designed to be readable by humans; if some
 * particular machine-readable format is needed, that should be
 * implemented outside the generic key-value tree code.
 *
 * \ingroup module_utility
 */
void dumpKeyValueTree(TextWriter* writer, const KeyValueTreeObject& tree);

/*! \brief
 * Compares two KeyValueTrees and prints any differences.
 *
 * \ingroup module_utility
 */
void compareKeyValueTrees(TextWriter*               writer,
                          const KeyValueTreeObject& tree1,
                          const KeyValueTreeObject& tree2,
                          real                      ftol,
                          real                      abstol);

//! Helper function to format a simple type within a KeyValueTreeValue.
std::string simpleValueToString(const KeyValueTreeValue& value);

//! \endcond

} // namespace gmx

#endif
