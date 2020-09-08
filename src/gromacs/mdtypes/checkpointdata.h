/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
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
 * \brief Provides the checkpoint data structure for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdtypes
 */

#ifndef GMX_MODULARSIMULATOR_CHECKPOINTDATA_H
#define GMX_MODULARSIMULATOR_CHECKPOINTDATA_H

#include <optional>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetreebuilder.h"

namespace gmx
{
class ISerializer;

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief The operations on CheckpointData
 *
 * This enum defines the two modes of operation on CheckpointData objects,
 * reading and writing. This allows to template all access functions, which
 * in turn enables clients to write a single function for read and write
 * access, eliminating the risk of having read and write functions getting
 * out of sync.
 */
enum class CheckpointDataOperation
{
    Read,
    Write,
    Count
};

/*! \internal
 * \ingroup module_modularsimulator
 * \brief Get an ArrayRef whose const-ness is defined by the checkpointing operation
 *
 * \tparam operation  Whether we are reading or writing
 * \tparam T          The type of values stored in the ArrayRef
 * \param container   The container the ArrayRef is referencing to
 * \return            The ArrayRef
 *
 * \see ArrayRef
 */
template<CheckpointDataOperation operation, typename T>
ArrayRef<std::conditional_t<operation == CheckpointDataOperation::Write || std::is_const<T>::value, const typename T::value_type, typename T::value_type>>
makeCheckpointArrayRef(T& container)
{
    return container;
}

/*! \internal
 * \ingroup module_modularsimulator
 * \brief Struct allowing to check if data is serializable through the KeyValueTree serializer
 *
 * This list of types is copied from ValueSerializer::initSerializers()
 * Having this here allows us to catch errors at compile time
 * instead of having cryptic runtime errors
 */
template<typename T>
struct IsSerializableType
{
    static bool const value = std::is_same<T, std::string>::value || std::is_same<T, bool>::value
                              || std::is_same<T, int>::value || std::is_same<T, int64_t>::value
                              || std::is_same<T, float>::value || std::is_same<T, double>::value;
};

/*! \internal
 * \ingroup module_modularsimulator
 * \brief Struct allowing to check if enum has a serializable underlying type
 */
//! {
template<typename T, bool = std::is_enum<T>::value>
struct IsSerializableEnum
{
    static bool const value = IsSerializableType<std::underlying_type_t<T>>::value;
};
template<typename T>
struct IsSerializableEnum<T, false>
{
    static bool const value = false;
};
//! }

/*! \libinternal
 * \ingroup module_modularsimulator
 * \brief Data type hiding checkpoint implementation details
 *
 * This data type allows to separate the implementation details of the
 * checkpoint writing / reading from the implementation of the checkpoint
 * clients. Checkpoint clients interface via the methods of the CheckpointData
 * object, and do not need knowledge of data types used to store the data.
 *
 * Templating allows checkpoint clients to have symmetric (templated)
 * implementations for checkpoint reading and writing.
 *
 * CheckpointData objects are dispatched via [Write|Read]CheckpointDataHolder
 * objects, which interact with the checkpoint reading from / writing to
 * file.
 */
template<CheckpointDataOperation operation>
class CheckpointData
{
public:
    /*! \brief Read or write a single value from / to checkpoint
     *
     * Allowed scalar types include std::string, bool, int, int64_t,
     * float, double, or any enum with one of the previously mentioned
     * scalar types as underlying type. Type compatibility is checked
     * at compile time.
     *
     * \tparam operation  Whether we are reading or writing
     * \tparam T          The type of the value
     * \param key         The key to [read|write] the value [from|to]
     * \param value       The value to [read|write]
     */
    //! {
    // Read
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
    scalar(const std::string& key, T* value) const;
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableEnum<T>::value, void>
    enumScalar(const std::string& key, T* value) const;
    // Write
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
    scalar(const std::string& key, const T* value);
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableEnum<T>::value, void>
    enumScalar(const std::string& key, const T* value);
    //! }

    /*! \brief Read or write an ArrayRef from / to checkpoint
     *
     * Allowed types stored in the ArrayRef include std::string, bool, int,
     * int64_t, float, double, and gmx::RVec. Type compatibility is checked
     * at compile time.
     *
     * \tparam operation  Whether we are reading or writing
     * \tparam T          The type of values stored in the ArrayRef
     * \param key         The key to [read|write] the ArrayRef [from|to]
     * \param values      The ArrayRef to [read|write]
     */
    //! {
    // Read ArrayRef of scalar
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
    arrayRef(const std::string& key, ArrayRef<T> values) const;
    // Write ArrayRef of scalar
    template<typename T, CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
    arrayRef(const std::string& key, ArrayRef<const T> values);
    // Read ArrayRef of RVec
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read, void> arrayRef(const std::string& key,
                                                                         ArrayRef<RVec> values) const;
    // Write ArrayRef of RVec
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write, void> arrayRef(const std::string& key,
                                                                          ArrayRef<const RVec> values);
    //! }

    /*! \brief Read or write a tensor from / to checkpoint
     *
     * \tparam operation  Whether we are reading or writing
     * \param key         The key to [read|write] the tensor [from|to]
     * \param values      The tensor to [read|write]
     */
    //! {
    // Read
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read, void> tensor(const std::string& key,
                                                                       ::tensor values) const;
    // Write
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write, void> tensor(const std::string& key,
                                                                        const ::tensor     values);
    //! }

    /*! \brief Return a subset of the current CheckpointData
     *
     * \tparam operation  Whether we are reading or writing
     * \param key         The key to [read|write] the sub data [from|to]
     * \return            A CheckpointData object representing a subset of the current object
     */
    //!{
    // Read
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Read, CheckpointData>
    subCheckpointData(const std::string& key) const;
    // Write
    template<CheckpointDataOperation op = operation>
    std::enable_if_t<op == CheckpointDataOperation::Write, CheckpointData>
    subCheckpointData(const std::string& key);
    //!}

private:
    //! KV tree read from checkpoint
    const KeyValueTreeObject* inputTree_ = nullptr;
    //! Builder for the tree to be written to checkpoint
    std::optional<KeyValueTreeObjectBuilder> outputTreeBuilder_ = std::nullopt;

    //! Construct an input checkpoint data object
    explicit CheckpointData(const KeyValueTreeObject& inputTree);
    //! Construct an output checkpoint data object
    explicit CheckpointData(KeyValueTreeObjectBuilder&& outputTreeBuilder);

    // Only holders should build
    friend class ReadCheckpointDataHolder;
    friend class WriteCheckpointDataHolder;
};

// Shortcuts
using ReadCheckpointData  = CheckpointData<CheckpointDataOperation::Read>;
using WriteCheckpointData = CheckpointData<CheckpointDataOperation::Write>;

/*! \libinternal
 * \brief Holder for read checkpoint data
 *
 * A ReadCheckpointDataHolder object is passed to the checkpoint reading
 * functionality, and then passed into the SimulatorBuilder object. It
 * holds the KV-tree read from file and dispatches CheckpointData objects
 * to the checkpoint clients.
 */
class ReadCheckpointDataHolder
{
public:
    //! Check whether a key exists
    [[nodiscard]] bool keyExists(const std::string& key) const;

    //! Return vector of existing keys
    [[nodiscard]] std::vector<std::string> keys() const;

    //! Deserialize serializer content into the CheckpointData object
    void deserialize(ISerializer* serializer);

    /*! \brief Return a subset of the current CheckpointData
     *
     * \param key         The key to [read|write] the sub data [from|to]
     * \return            A CheckpointData object representing a subset of the current object
     */
    [[nodiscard]] ReadCheckpointData checkpointData(const std::string& key) const;

private:
    //! KV-tree read from checkpoint
    KeyValueTreeObject checkpointTree_;
};

/*! \libinternal
 * \brief Holder for write checkpoint data
 *
 * The WriteCheckpointDataHolder object holds the KV-tree builder and
 * dispatches CheckpointData objects to the checkpoint clients to save
 * their respective data. It is then passed to the checkpoint writing
 * functionality.
 */
class WriteCheckpointDataHolder
{
public:
    //! Serialize the content of the CheckpointData object
    void serialize(ISerializer* serializer);

    /*! \brief Return a subset of the current CheckpointData
     *
     * \param key         The key to [read|write] the sub data [from|to]
     * \return            A CheckpointData object representing a subset of the current object
     */
    [[nodiscard]] WriteCheckpointData checkpointData(const std::string& key);

    /*! \brief
     */
    [[nodiscard]] bool empty() const;

private:
    //! KV-tree builder
    KeyValueTreeBuilder outputTreeBuilder_;
    //! Whether any checkpoint data object has been requested
    bool hasCheckpointDataBeenRequested_ = false;
};

// Function definitions - here to avoid template-related linker problems
// doxygen doesn't like these...
//! \cond
template<>
template<typename T, CheckpointDataOperation op>
std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
ReadCheckpointData::scalar(const std::string& key, T* value) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    *value = (*inputTree_)[key].cast<T>();
}

template<>
template<typename T, CheckpointDataOperation op>
std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableEnum<T>::value, void>
ReadCheckpointData::enumScalar(const std::string& key, T* value) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    std::underlying_type_t<T> castValue;
    castValue = (*inputTree_)[key].cast<std::underlying_type_t<T>>();
    *value    = static_cast<T>(castValue);
}

template<>
template<typename T, CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
WriteCheckpointData::scalar(const std::string& key, const T* value)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    outputTreeBuilder_->addValue(key, *value);
}

template<>
template<typename T, CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableEnum<T>::value, void>
WriteCheckpointData::enumScalar(const std::string& key, const T* value)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto castValue = static_cast<std::underlying_type_t<T>>(*value);
    outputTreeBuilder_->addValue(key, castValue);
}

template<>
template<typename T, CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Read && IsSerializableType<T>::value, void>
ReadCheckpointData::arrayRef(const std::string& key, ArrayRef<T> values) const
{
    GMX_RELEASE_ASSERT(inputTree_, "No input checkpoint data available.");
    GMX_RELEASE_ASSERT(values.size() >= (*inputTree_)[key].asArray().values().size(),
                       "Read vector does not fit in passed ArrayRef.");
    auto outputIt  = values.begin();
    auto inputIt   = (*inputTree_)[key].asArray().values().begin();
    auto outputEnd = values.end();
    auto inputEnd  = (*inputTree_)[key].asArray().values().end();
    for (; outputIt != outputEnd && inputIt != inputEnd; outputIt++, inputIt++)
    {
        *outputIt = inputIt->cast<T>();
    }
}

template<>
template<typename T, CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write && IsSerializableType<T>::value, void>
WriteCheckpointData::arrayRef(const std::string& key, ArrayRef<const T> values)
{
    GMX_RELEASE_ASSERT(outputTreeBuilder_, "No output checkpoint data available.");
    auto builder = outputTreeBuilder_->addUniformArray<T>(key);
    for (const auto& value : values)
    {
        builder.addValue(value);
    }
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Read, void>
ReadCheckpointData::arrayRef(const std::string& key, ArrayRef<RVec> values) const
{
    GMX_RELEASE_ASSERT(values.size() >= (*inputTree_)[key].asArray().values().size(),
                       "Read vector does not fit in passed ArrayRef.");
    auto outputIt  = values.begin();
    auto inputIt   = (*inputTree_)[key].asArray().values().begin();
    auto outputEnd = values.end();
    auto inputEnd  = (*inputTree_)[key].asArray().values().end();
    for (; outputIt != outputEnd && inputIt != inputEnd; outputIt++, inputIt++)
    {
        auto storedRVec = inputIt->asObject()["RVec"].asArray().values();
        *outputIt       = { storedRVec[XX].cast<real>(), storedRVec[YY].cast<real>(),
                      storedRVec[ZZ].cast<real>() };
    }
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write, void>
WriteCheckpointData::arrayRef(const std::string& key, ArrayRef<const RVec> values)
{
    auto builder = outputTreeBuilder_->addObjectArray(key);
    for (const auto& value : values)
    {
        auto subbuilder = builder.addObject();
        subbuilder.addUniformArray("RVec", { value[XX], value[YY], value[ZZ] });
    }
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Read, void>
ReadCheckpointData::tensor(const std::string& key, ::tensor values) const
{
    auto array     = (*inputTree_)[key].asArray().values();
    values[XX][XX] = array[0].cast<real>();
    values[XX][YY] = array[1].cast<real>();
    values[XX][ZZ] = array[2].cast<real>();
    values[YY][XX] = array[3].cast<real>();
    values[YY][YY] = array[4].cast<real>();
    values[YY][ZZ] = array[5].cast<real>();
    values[ZZ][XX] = array[6].cast<real>();
    values[ZZ][YY] = array[7].cast<real>();
    values[ZZ][ZZ] = array[8].cast<real>();
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write, void>
WriteCheckpointData::tensor(const std::string& key, const ::tensor values)
{
    auto builder = outputTreeBuilder_->addUniformArray<real>(key);
    builder.addValue(values[XX][XX]);
    builder.addValue(values[XX][YY]);
    builder.addValue(values[XX][ZZ]);
    builder.addValue(values[YY][XX]);
    builder.addValue(values[YY][YY]);
    builder.addValue(values[YY][ZZ]);
    builder.addValue(values[ZZ][XX]);
    builder.addValue(values[ZZ][YY]);
    builder.addValue(values[ZZ][ZZ]);
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Read, ReadCheckpointData>
ReadCheckpointData::subCheckpointData(const std::string& key) const
{
    return CheckpointData((*inputTree_)[key].asObject());
}

template<>
template<CheckpointDataOperation op>
inline std::enable_if_t<op == CheckpointDataOperation::Write, WriteCheckpointData>
WriteCheckpointData::subCheckpointData(const std::string& key)
{
    return CheckpointData(outputTreeBuilder_->addObject(key));
}

template<CheckpointDataOperation operation>
CheckpointData<operation>::CheckpointData(const KeyValueTreeObject& inputTree) :
    inputTree_(&inputTree)
{
    static_assert(operation == CheckpointDataOperation::Read,
                  "This constructor can only be called for a read CheckpointData");
}

template<CheckpointDataOperation operation>
CheckpointData<operation>::CheckpointData(KeyValueTreeObjectBuilder&& outputTreeBuilder) :
    outputTreeBuilder_(outputTreeBuilder)
{
    static_assert(operation == CheckpointDataOperation::Write,
                  "This constructor can only be called for a write CheckpointData");
}
//! \endcond

} // namespace gmx

#endif // GMX_MODULARSIMULATOR_CHECKPOINTDATA_H
