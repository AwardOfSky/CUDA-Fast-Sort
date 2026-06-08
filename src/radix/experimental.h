/*
    Experimental header

    Features:
        Reflection Prototyping
        (To be used to support AoS API calls when NVCC gets reflection)
*/
#pragma once


// check for reflection capabilities
#if !defined(__CUDACC__) && \
    defined(__cpp_impl_reflection) && (__cpp_impl_reflection >= 202506L)
    #define RSORT_CPP_REFLECTION 1
#else
    #define RSORT_CPP_REFLECTION 0
#endif


// experimental C++ reflection prototype (tested with g++-16 - 16.0.1 20260315)
#if RSORT_CPP_REFLECTION

#include <cstdlib>
#include <cstdint>
#include <typeinfo>
#include <meta>


namespace rsort {

// quick fixed_string implementation
template<size_t N>
struct fixed_string {
    char data[N];

    // initializers 
    constexpr fixed_string(const char (&s)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = s[i];
        }
    }

    // operators
    constexpr bool operator==(const fixed_string&) const = default;
    
    constexpr char operator[](size_t i) const {
        return data[i];
    }

    // std functions
    constexpr const char* c_str() const {
        return data;
    }

    constexpr std::size_t size() const {
        return N ? N - 1 : 0;
    }

    constexpr std::string_view view() const {
        return {data, size()};
    }
};


// get key machinery
template<typename T, fixed_string Name>
consteval auto reflection_find_member() {
    constexpr auto ctx = std::meta::access_context::current();

    for (auto m : std::meta::nonstatic_data_members_of(^^T, ctx)) {
        if (std::meta::identifier_of(m) == Name.view()) {
            return m;
        }
    }
    return std::meta::info{};
}

template<typename T>
using ptr_t = T*;


/*
    Transform AoS:
        KV Pair  [Array]
            key
            member_a
            member_b
            member_c
            ...
    
    Into SoA:
        SoA Type
            key [array]
            value [array]
                member_a
                member_b
                member_c
                ...   
*/
template<typename Pair, fixed_string Key>
struct soa_traits {
    struct value_type;
    struct soa_type;


    static consteval std::meta::info get_key() {
        return reflection_find_member<Pair, Key>();
    }

    consteval {
        constexpr auto ctx = std::meta::access_context::current();
        constexpr std::meta::info key_meta = get_key();

        std::vector<std::meta::info> value_fields;        

        for (std::meta::info m : std::meta::nonstatic_data_members_of(^^Pair, ctx)) {
            if (std::meta::identifier_of(m) != Key.view()) {   
                value_fields.push_back(
                    std::meta::data_member_spec(
                        std::meta::type_of(m),
                        {.name = std::meta::identifier_of(m)}
                    )
                );
            }
        }

        static_assert(
            key_meta != std::meta::info{},
            "[soa_traits]: key name not found in type!"
        );

        std::meta::define_aggregate(^^value_type, value_fields);

        std::vector<std::meta::info> soa_fields = {
            std::meta::data_member_spec(
                ^^value_type*,
                {.name = "value"}
            ),
            std::meta::data_member_spec(
                std::meta::substitute(^^ptr_t, {std::meta::type_of(key_meta)}),
                {.name = std::meta::identifier_of(key_meta)}
            )
        };

        std::meta::define_aggregate(^^soa_type, soa_fields);
    }
};

} // namespace rsort

#endif // RSORT_CPP_REFLECTION
