#include <string>

class Line
{
private:
    int _line_number{};
    std::string _raw_input{};
    uint16_t _binary{0};

public:
    Line(std::string input_line);
    ~Line();

    std::string getRawInput()
    uint16_t getBinaryRepresentation();
    int getLineNumber();
    int getLineAddress();
    

    const uint16_t JUMP_NULL{0};
    const uint16_t JUMP_JGT{1};
    const uint16_t JUMP_JEQ{2};
    const uint16_t JUMP_JGE{3};
    const uint16_t JUMP_JLT{4};
    const uint16_t JUMP_JNE{5};
    const uint16_t JUMP_JLE{6};
    const uint16_t JUMP_JMP{7};

    const uint16_t DEST_NULL{0 << 3};
    const uint16_t DEST_M{1 << 3};
    const uint16_t DEST_D{2 << 3};
    const uint16_t DEST_MD{3 << 3};
    const uint16_t DEST_A{4 << 3};
    const uint16_t DEST_AM{5 << 3};
    const uint16_t DEST_AD{6 << 3};
    const uint16_t DEST_AMD{7 << 3};
};

