#pragma once

#include "ParallelPhysics/ServerProtocol.h"
#include <array>
#include <atomic>
#include <vector>
#include <memory>

class ConditionedReflexNeuron;
class PrognosticNeuron;
class Neuron;

constexpr uint32_t SECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND;  // quantum of time
constexpr uint32_t MILLISECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000;  // quantum of time

class Synapse
{
public:
	Synapse(Neuron *from);
	uint32_t Tick() const;
private:
	Neuron *m_from;
};

typedef std::vector<Synapse> SynapseVector;
typedef std::shared_ptr<SynapseVector> SP_SynapseVector;

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

	virtual void Init() {}
	virtual void Tick() {}

	virtual bool IsActive() const;
	virtual uint32_t ReadAxon() const { return 0; }

	virtual uint8_t GetType() = 0;
	uint32_t GetId();

protected:
	enum class NeuronTypes
	{
		None = 0,
		SensoryNeuron,
		SensoryNeuronRed,
		SensoryNeuronGreen,
		SensoryNeuronBlue,
		MotorNeuron,
		GeneralizingNeuron
	};
};

class SensoryNeuron : public Neuron
{
public:
	SensoryNeuron() = default;
	virtual ~SensoryNeuron() = default;

	void Init()  override;

	bool IsActive() const override;
	uint32_t ReadAxon() const override;
	void WriteDendrite(uint8_t exitation);
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;


private:
	uint8_t m_dendrite[2];
	uint8_t m_axon[2];
};

class SensoryNeuronRed : public SensoryNeuron
{
public:
	SensoryNeuronRed() = default;
	virtual ~SensoryNeuronRed() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronRed); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
};

class SensoryNeuronGreen : public SensoryNeuron
{
public:
	SensoryNeuronGreen() = default;
	virtual ~SensoryNeuronGreen() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronGreen); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
};

class SensoryNeuronBlue : public SensoryNeuron
{
public:
	SensoryNeuronBlue() = default;
	virtual ~SensoryNeuronBlue() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronBlue); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	bool IsActive() const override;
};

class MotorNeuron : public Neuron
{
public:
	MotorNeuron() = default;
	virtual ~MotorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init() override;

	bool IsActive() const override;

	void Tick() override;

	void ExcitatorySynapse();
	static uint32_t GetMovingSpontaneousCount();
private:
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	uint16_t m_accumulatedExcitation;
	uint64_t m_lastExcitationTime;
	uint32_t m_spontaneusActivityTimeStart;
	uint64_t m_spontaneusActivityTimeFinishAbs;
	bool m_isActive[2];
	static std::atomic<uint32_t> m_movingSpontaneousCount;
};

class GeneralizingNeuron : public Neuron
{
public:
	GeneralizingNeuron();
	void Init(const SP_SynapseVector &synapses);
	virtual ~GeneralizingNeuron() = default;

	uint32_t ReadAxon() const override;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::GeneralizingNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;


private:
	uint16_t m_axon[2];
	SP_SynapseVector m_synapses;
};
