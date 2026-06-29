require "gnuplot"

class Noh
  def self.draw
    Gnuplot.open do |gp|
      Gnuplot::SPlot.new(gp) do |plot|
        plot.terminal "jpeg"
        plot.output "plot/noh.jpeg"
        plot.xlabel "x"
        plot.ylabel "y"
        plot.unset "key"
        plot.set "contour"
        plot.set "cntrparam levels 100"
        plot.set "view 0,0"
        plot.unset "surface"
        plot.set "size square"

        plot.data << Gnuplot::DataSet.new("'dat/noh.dat' matrix") do |ds|
          ds.with = "lines"
        end
      end
    end
  end
end
