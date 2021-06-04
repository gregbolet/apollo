
class CaliperCounters : public PerfCounter{

    public:
        void CaliperCounters();
        void ~CaliperCounters();

        void start();
        void stop();        
        void query();
        
}