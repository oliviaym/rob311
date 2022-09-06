/*
 * Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// -----------------------------------------------------
// NOTE: THIS HEADER IS ALSO INCLUDED BY ASSEMBLER SO
//       SHOULD ONLY CONSIST OF PREPROCESSOR DIRECTIVES
// -----------------------------------------------------

#ifndef _BOARDS_PIMORONI_BADGER2040_H
#define _BOARDS_PIMORONI_BADGER2040_H

// For board detection
#define PIMORONI_BADGER2040

// --- BOARD SPECIFIC ---
#define BADGER2040_UART 0
#define BADGER2040_TX_PIN 0
#define BADGER2040_RX_PIN 1

#define BADGER2040_I2C 0
#define BADGER2040_INT_PIN 3
#define BADGER2040_SDA_PIN 4
#define BADGER2040_SCL_PIN 5

#define BADGER2040_3V3_EN_PIN 10

#define BADGER2040_SW_DOWN_PIN 11
#define BADGER2040_SW_A_PIN 12
#define BADGER2040_SW_B_PIN 13
#define BADGER2040_SW_C_PIN 14
#define BADGER2040_SW_UP_PIN 15

#define BADGER2040_INKY_SPI 0
#define BADGER2040_INKY_MISO_PIN 16
#define BADGER2040_INKY_CSN_PIN 17
#define BADGER2040_INKY_SCK_PIN 18
#define BADGER2040_INKY_MOSI_PIN 19
#define BADGER2040_INKY_DC_PIN 20
#define BADGER2040_INKY_RESET_PIN 21
#define BADGER2040_INKY_BUSY_PIN 26

#define BADGER2040_USER_SW_PIN 23
#define BADGER2040_USER_LED_PIN 25

#define BADGER2040_VBUS_DETECT_PIN 24
#define BADGER2040_VREF_POWER_PIN 27
#define BADGER2040_1V2_REF_PIN 28
#define BADGER2040_BAT_SENSE_PIN 29

// --- UART ---
#ifndef PICO_DEFAULT_UART
#define PICO_DEFAULT_UART BADGER2040_UART
#endif

#ifndef PICO_DEFAULT_UART_TX_PIN
#define PICO_DEFAULT_UART_TX_PIN BADGER2040_TX_PIN
#endif

#ifndef PICO_DEFAULT_UART_RX_PIN
#define PICO_DEFAULT_UART_RX_PIN BADGER2040_RX_PIN
#endif

// --- LED ---
#ifndef PICO_DEFAULT_LED_PIN
#define PICO_DEFAULT_LED_PIN BADGER2040_USER_LED_PIN
#endif
// no PICO_DEFAULT_WS2812_PIN

// --- I2C ---
#ifndef PICO_DEFAULT_I2C
#define PICO_DEFAULT_I2C BADGER2040_I2C
#endif
#ifndef PICO_DEFAULT_I2C_SDA_PIN
#define PICO_DEFAULT_I2C_SDA_PIN BADGER2040_SDA_PIN
#endif
#ifndef PICO_DEFAULT_I2C_SCL_PIN
#define PICO_DEFAULT_I2C_SCL_PIN BADGER2040_SCL_PIN
#endif

// --- SPI ---
#ifndef PICO_DEFAULT_SPI
#define PICO_DEFAULT_SPI BADGER2040_INKY_SPI
#endif
#ifndef PICO_DEFAULT_SPI_SCK_PIN
#define PICO_DEFAULT_SPI_SCK_PIN BADGER2040_INKY_SCK_PIN
#endif
#ifndef PICO_DEFAULT_SPI_TX_PIN
#define PICO_DEFAULT_SPI_TX_PIN BADGER2040_INKY_MOSI_PIN
#endif
#ifndef PICO_DEFAULT_SPI_RX_PIN
#define PICO_DEFAULT_SPI_RX_PIN BADGER2040_INKY_MISO_PIN
#endif
#ifndef PICO_DEFAULT_SPI_CSN_PIN
#define PICO_DEFAULT_SPI_CSN_PIN BADGER2040_INKY_CSN_PIN
#endif

// --- FLASH ---
#define PICO_BOOT_STAGE2_CHOOSE_W25Q080 1

#ifndef PICO_FLASH_SPI_CLKDIV
#define PICO_FLASH_SPI_CLKDIV 2
#endif

#ifndef PICO_FLASH_SIZE_BYTES
#define PICO_FLASH_SIZE_BYTES (2 * 1024 * 1024)
#endif

// All boards have B1 RP2040
#ifndef PICO_RP2040_B0_SUPPORTED
#define PICO_RP2040_B0_SUPPORTED 0
#endif

#endif
