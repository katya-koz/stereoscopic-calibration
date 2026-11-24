#pragma once

// Calibration state machine:
//
//   StartInstructions --(A / Enter)--> HeadPositioning
//   HeadPositioning --(A / Enter, only while the guide is green)--> Calibration
//   HeadPositioning --(B / Backspace)--> StartInstructions
//   Calibration --(all points collected)--> Calculating
//   Calculating --> Complete | Failure
//   Failure --(A / Enter)--> StartInstructions
//   Complete --(B / Backspace)--> StartInstructions
//   Complete --(A / Enter)--> [write csv] --> Done
enum class CalibrationStage {
	StartInstructions,
	HeadPositioning,
	Calibration,
	Calculating,
	Complete,
	Failure,
	Done
};